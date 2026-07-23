#!/usr/bin/env bash
# One-time Kaggle CLI auth setup.
# Get kaggle.json from: kaggle.com -> Settings -> API -> "Create New Token"
# Then either drop that file at ~/.kaggle/kaggle.json, or run:
#   ./scripts/setup_kaggle_auth.sh <username> <api_key>
set -euo pipefail

mkdir -p "$HOME/.kaggle"

if [[ $# -eq 2 ]]; then
  printf '{"username":"%s","key":"%s"}\n' "$1" "$2" > "$HOME/.kaggle/kaggle.json"
  echo "Wrote ~/.kaggle/kaggle.json"
elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "Found existing ~/.kaggle/kaggle.json"
else
  echo "ERROR: no kaggle.json. Pass '<username> <key>' or place the file yourself." >&2
  exit 1
fi

chmod 600 "$HOME/.kaggle/kaggle.json"
echo "Verifying..."
python3 -m kaggle datasets list -s titanic | head -3
echo "Kaggle CLI authenticated."
