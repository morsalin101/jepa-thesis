"""Frozen-feature evaluation: k-NN and linear probing on HyperKvasir-labelled.

The cheapest signal about representation quality that exists — no decoder, no
fine-tuning, ~20 minutes for all four encoders. Run it as soon as pretraining finishes
and *before* committing GPU hours to segmentation: if an encoder's k-NN accuracy is at
chance, something is wrong with the pretraining run and no amount of fine-tuning will
hide it.

Note this uses the labelled HyperKvasir split (10,662 images, 23 classes), which is never
trained on anywhere in this project. It is a diagnostic, not a headline result.

    python -m src.eval.probe --encoders ijepa mae simclr mocov3 random
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import amp_config, default_output_dir, resolve_dataset_dir
from src.data.hyperkvasir import HyperKvasirLabeled
from src.data.transforms import make_pretrain_transform
from src.model.vit import build_vit


@torch.no_grad()
def extract_features(
    encoder: torch.nn.Module, loader: DataLoader, device: str, amp
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pooled patch tokens for every image. No augmentation, model in eval mode."""
    encoder.eval()
    feats, labels = [], []
    for i, (imgs, y) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda" if str(device).startswith("cuda") else "cpu",
            dtype=amp.dtype or torch.float32,
            enabled=amp.enabled,
        ):
            f = encoder(imgs).mean(dim=1)
        feats.append(f.float().cpu())
        labels.append(y)
        if i % 20 == 0:
            print(f"[probe]   batch {i}/{len(loader)}")
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def knn_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    k: int = 20,
    temperature: float = 0.07,
) -> float:
    """Weighted k-NN on L2-normalised features, as used by DINO/MoCo evaluations."""
    tr = F.normalize(torch.from_numpy(train_x), dim=1)
    te = F.normalize(torch.from_numpy(test_x), dim=1)
    tr_y = torch.from_numpy(train_y)
    n_classes = int(tr_y.max()) + 1

    correct = 0
    for start in range(0, len(te), 256):
        chunk = te[start : start + 256]
        sim = chunk @ tr.T
        top_sim, top_idx = sim.topk(min(k, tr.shape[0]), dim=1)
        weights = (top_sim / temperature).exp()
        votes = torch.zeros(chunk.shape[0], n_classes)
        votes.scatter_add_(1, tr_y[top_idx], weights)
        correct += int((votes.argmax(dim=1).numpy() == test_y[start : start + 256]).sum())
    return correct / len(te)


def linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> float:
    """Logistic regression on frozen features, trained with AdamW."""
    mu, sd = train_x.mean(0, keepdims=True), train_x.std(0, keepdims=True) + 1e-6
    tx = torch.from_numpy((train_x - mu) / sd).float().to(device)
    ty = torch.from_numpy(train_y).long().to(device)
    vx = torch.from_numpy((test_x - mu) / sd).float().to(device)

    clf = torch.nn.Linear(tx.shape[1], int(train_y.max()) + 1).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(clf(tx), ty).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pred = clf(vx).argmax(dim=1).cpu().numpy()
    return float((pred == test_y).mean())


def load_encoder(weights_dir: Path, encoder: str, arch: str, img_size: int, patch_size: int):
    """Build a ViT and load exported SSL weights. `random` returns an untrained model."""
    model = build_vit(arch, img_size=img_size, patch_size=patch_size)
    if encoder == "random":
        return model, None
    matches = sorted(weights_dir.glob(f"{encoder}_{arch}_*.pt"))
    if not matches:
        raise FileNotFoundError(f"no exported encoder for {encoder!r} under {weights_dir}")
    ckpt = torch.load(matches[-1], map_location="cpu", weights_only=False)
    state = {k: v for k, v in ckpt["encoder"].items() if not k.endswith("pos_embed")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith("pos_embed")]
    if real_missing or unexpected:
        raise RuntimeError(f"{encoder}: mismatched weights (missing={real_missing[:3]})")
    return model, matches[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="k-NN and linear probe on frozen features")
    ap.add_argument("--encoders", nargs="+", default=["ijepa", "mae", "simclr", "mocov3", "random"])
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-embeddings", action="store_true", help="also write a UMAP projection")
    args = ap.parse_args()

    amp = amp_config()
    device = "cuda:0" if amp.device == "cuda" else amp.device
    base = default_output_dir()
    out_path = Path(args.out) if args.out else base / "results" / "probe.json"

    root = Path(args.data_root) if args.data_root else resolve_dataset_dir("hyperkvasir_labeled")
    tf = make_pretrain_transform(args.img_size, crop_scale=(1.0, 1.0), horizontal_flip=False)
    ds = HyperKvasirLabeled(root, transform=tf)
    print(f"[probe] {len(ds)} labelled images, {len(ds.classes)} classes")

    # Fixed split, so every encoder is scored on exactly the same images.
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(ds))
    n_train = int(len(ds) * args.train_frac)
    idx_train, idx_test = perm[:n_train], perm[n_train:]

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    results: dict[str, dict] = {}
    if out_path.is_file():
        results = json.loads(out_path.read_text())
    embeddings: dict[str, np.ndarray] = {}

    for enc in args.encoders:
        print(f"\n[probe] === {enc} ===")
        model, src = load_encoder(base / "weights", enc, args.arch, args.img_size, args.patch_size)
        model = model.to(device)
        feats, labels = extract_features(model, loader, device, amp)

        knn = knn_accuracy(feats[idx_train], labels[idx_train], feats[idx_test], labels[idx_test], args.k)
        lin = linear_probe(feats[idx_train], labels[idx_train], feats[idx_test], labels[idx_test], device=device)
        results[enc] = {
            "knn_top1": knn,
            "linear_top1": lin,
            "k": args.k,
            "n_train": len(idx_train),
            "n_test": len(idx_test),
            "n_classes": len(ds.classes),
            "weights": str(src) if src else None,
        }
        print(f"[probe] {enc}: k-NN top-1 {knn:.2%}   linear top-1 {lin:.2%}")

        if args.save_embeddings:
            embeddings[enc] = feats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[probe] wrote {out_path}")

    if args.save_embeddings and embeddings:
        proj: dict[str, np.ndarray] = {"labels": labels}
        for enc, f in embeddings.items():
            try:
                import umap

                xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(f)
            except ImportError:
                from sklearn.decomposition import PCA

                print("[probe] umap-learn not installed; falling back to PCA for the projection")
                xy = PCA(n_components=2, random_state=0).fit_transform(f)
            proj[f"{enc}_xy"] = np.asarray(xy)
        emb_path = out_path.parent / "embeddings.npz"
        np.savez_compressed(emb_path, **proj)
        print(f"[probe] wrote {emb_path}")


if __name__ == "__main__":
    main()
