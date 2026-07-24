#!/usr/bin/env bash
# Resolves the kaggle CLI across installs (system pip, pip --user, brew, etc.).
# Tries in order: `kaggle` on PATH, common pip --user bins, then python -m fallback.
kaggle_cmd() {
  if command -v kaggle >/dev/null 2>&1; then
    command -v kaggle
    return 0
  fi
  for cand in \
    "$HOME/Library/Python/3.9/bin/kaggle" \
    "$HOME/Library/Python/3.8/bin/kaggle" \
    "$HOME/Library/Python/3.11/bin/kaggle" \
    "$HOME/Library/Python/3.12/bin/kaggle" \
    "$HOME/Library/Python/3.13/bin/kaggle" \
    "$HOME/.local/bin/kaggle"
  do
    if [[ -x "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done
  if python3 -c "import kaggle" >/dev/null 2>&1 && python3 -c "import importlib.util; importlib.util.find_spec('kaggle.__main__')" >/dev/null 2>&1; then
    echo "python3 -m kaggle"
    return 0
  fi
  echo "ERROR: kaggle CLI not found. Run: python3 -m pip install --user kaggle" >&2
  return 1
}
