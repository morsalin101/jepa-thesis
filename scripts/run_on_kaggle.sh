#!/usr/bin/env bash
# Push the runner notebook to Kaggle and kick off a GPU run, headless.
# Usage: ./scripts/run_on_kaggle.sh
#
# The notebook itself does `git pull` from GitHub, so the flow is:
#   1) git push your code changes to GitHub
#   2) run this script -> Kaggle pulls latest + runs on GPU
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../notebooks" && pwd)"
KERNEL_ID="$(python3 -c "import json; print(json.load(open('$DIR/kernel-metadata.json'))['id'])")

source "$(dirname "${BASH_SOURCE[0]}")/_kaggle_resolve.sh"
KG="$(kaggle_cmd)"

echo ">> Pushing kernel $KERNEL_ID ..."
$KG kernels push -p "$DIR"

echo ">> Started. Watch status with:"
echo "   $KG kernels status $KERNEL_ID"
echo ">> Fetch outputs when complete:"
echo "   $KG kernels output $KERNEL_ID -p ./outputs"
