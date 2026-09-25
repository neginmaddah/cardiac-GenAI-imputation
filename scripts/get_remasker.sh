#!/usr/bin/env bash
# Fetch ReMasker (https://github.com/alps-lab/remasker) at the commit the
# published results used, and apply the one-line NumPy compatibility fix.
# ReMasker is not redistributed here because upstream carries no licence.
# See docs/remasker.md.
set -euo pipefail

REPO=https://github.com/alps-lab/remasker.git
COMMIT=b2eca0eed92a03cedca9a19b3c42f6d7da7231bb
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/external/remasker"

if [ ! -d "$DEST/.git" ]; then
    git clone --quiet "$REPO" "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"

# np.float was removed in NumPy 1.24; this project pins 1.26.4.
sed -i.bak 's/dtype=np\.float)/dtype=float)/' "$DEST/utils.py" && rm -f "$DEST/utils.py.bak"

if grep -q 'np\.float)' "$DEST/utils.py"; then
    echo "error: NumPy patch did not apply to $DEST/utils.py" >&2
    exit 1
fi
echo "ReMasker ready at $DEST"
