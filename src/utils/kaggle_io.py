"""Persisting state between Kaggle sessions.

`/kaggle/working` is wiped when a session ends, so a checkpoint that only lives there is
lost the moment the 9-hour cap hits. Three ways out, and they are not equivalent:

* **A Kaggle Dataset we own, versioned from inside the kernel.** Push-based, so it can
  happen *mid-run*. A crash at hour 7.4 costs one push interval, not the session.
* **The previous kernel version's committed output** (`kernel_sources` pointing at the
  notebook itself). Free and automatic, but only produced when a commit *succeeds* — and
  a run that exceeds the time limit fails, taking its output with it. Good mirror,
  unreliable primary.
* **Kaggle Models.** Built for release artefacts; too heavy for a 400 MB blob rewritten
  every 45 minutes. Used here only for the four final encoders.

So: dataset push for hot resume state, kernel output as a free mirror, and a session
guard that exits *cleanly* before the platform kills us so the commit succeeds.

Credentials come from Kaggle Secrets (Add-ons -> Secrets), never a committed
`kaggle.json`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from src.config import on_kaggle
from src.utils.ddp import is_main


def load_kaggle_credentials(verbose: bool = True) -> bool:
    """Populate KAGGLE_USERNAME / KAGGLE_KEY from Kaggle Secrets.

    Returns False if credentials are unavailable — callers should degrade to local-only
    checkpointing rather than crash, since a run without dataset pushes is still a run.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        secrets = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
        return True
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(
                f"[kaggle_io] no credentials ({e}). Checkpoints stay in /kaggle/working "
                "only and will be lost when the session ends. Add KAGGLE_USERNAME and "
                "KAGGLE_KEY under Add-ons -> Secrets to enable cross-session resume."
            )
        return False


def _run(cmd: list[str], timeout: float = 900.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class CheckpointPusher:
    """Versions a directory into a Kaggle Dataset on a time-based cadence.

    Cadence is time-based rather than epoch-based on purpose: epochs range from 2.7 min
    (MAE) to 12.8 min (MoCo v3), so "every N epochs" would mean wildly different
    exposure to a crash across methods. ~45 min caps the loss uniformly at ~10% of a
    session for roughly 4% overhead.
    """

    def __init__(
        self,
        slug: str,
        staging_dir: str | os.PathLike[str],
        title: str | None = None,
        push_minutes: float = 45.0,
        enabled: bool = True,
    ) -> None:
        self.slug = slug
        self.dir = Path(staging_dir)
        self.title = title or (slug.split("/")[-1] if slug else "checkpoints")
        self.push_interval = push_minutes * 60.0
        self.last_push = time.time()
        self.enabled = bool(enabled and slug and on_kaggle() and is_main())
        if self.enabled:
            self.enabled = load_kaggle_credentials()
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._write_metadata()

    def _write_metadata(self) -> None:
        (self.dir / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": self.title,
                    "id": self.slug,
                    "licenses": [{"name": "CC0-1.0"}],
                },
                indent=2,
            )
        )

    def due(self) -> bool:
        return self.enabled and (time.time() - self.last_push) >= self.push_interval

    def push(self, message: str, force: bool = False) -> bool:
        """Create a new dataset version. Returns True on success.

        Never raises: a failed push (flaky network, quota) must not kill a training run
        that is otherwise healthy. The next push attempt will carry the same state.
        """
        if not self.enabled or (not force and not self.due()):
            return False
        self.last_push = time.time()
        cmd = [
            "kaggle", "datasets", "version",
            "-p", str(self.dir),
            "-m", message[:500],
            "--dir-mode", "zip",
            "--quiet",
        ]
        try:
            r = _run(cmd)
        except subprocess.TimeoutExpired:
            print("[kaggle_io] push timed out; continuing training")
            return False
        if r.returncode != 0:
            err = (r.stderr or r.stdout).strip().splitlines()
            if err and "does not exist" in err[-1].lower():
                print(
                    f"[kaggle_io] dataset {self.slug} does not exist yet. Create it once "
                    f"with:  kaggle datasets create -p {self.dir} --dir-mode zip --private"
                )
            else:
                print(f"[kaggle_io] push failed: {err[-1] if err else r.returncode}")
            return False
        print(f"[kaggle_io] pushed {self.slug}: {message}")
        return True


def resume_candidates(
    filename: str,
    ckpt_slug: str = "",
    kernel_slug: str = "",
    working_dir: str | os.PathLike[str] = "/kaggle/working",
) -> list[Path]:
    """Ordered places to look for a resume checkpoint, most-recent-first."""
    out: list[Path] = []
    if ckpt_slug:
        out.append(Path("/kaggle/input") / ckpt_slug.split("/")[-1] / filename)
    if kernel_slug:
        base = Path("/kaggle/input") / kernel_slug.split("/")[-1]
        out += [base / filename, base / "ckpt" / filename]
    out.append(Path(working_dir) / filename)
    out.append(Path(working_dir) / "ckpt" / filename)
    return out


class SessionGuard:
    """Stops training before Kaggle kills the session.

    Exiting cleanly matters more than it sounds: a notebook that runs past the platform
    limit is marked *failed*, and a failed commit produces no output — so the free
    `kernel_sources` mirror disappears exactly when it would have been most useful.
    Break out of the loop, save, push, and let the commit finish.
    """

    def __init__(self, hours: float = 7.5) -> None:
        self.start = time.time()
        self.limit = hours * 3600.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.elapsed)

    def expired(self, margin_s: float = 0.0) -> bool:
        return self.elapsed + margin_s >= self.limit

    def would_exceed(self, next_epoch_s: float) -> bool:
        """True if another epoch probably would not finish before the guard fires."""
        return self.elapsed + next_epoch_s >= self.limit

    def summary(self) -> str:
        h, rem = divmod(int(self.elapsed), 3600)
        return f"{h}h{rem // 60:02d}m elapsed, {self.remaining / 3600:.1f}h before guard"


def stage_file(src: str | os.PathLike[str], staging_dir: str | os.PathLike[str]) -> Path:
    """Copy a file into the push staging directory (same-filesystem move if possible)."""
    src, staging = Path(src), Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    dst = staging / src.name
    if src.resolve() == dst.resolve():
        return dst
    shutil.copy2(src, dst)
    return dst


def append_metrics(path: str | os.PathLike[str], record: dict) -> None:
    """Append one JSON line of metrics.

    Every figure script reads these files rather than a checkpoint, so figures can be
    iterated on a laptop with no GPU and no 400 MB download.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
