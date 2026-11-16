#!/bin/zsh
set -eu
LEAFCHECK_DIR="${0:A:h}"
LEAFCHECK_PYTHON=""

# Prefer an app-local environment, then the existing notebook environment.
for candidate in "$LEAFCHECK_DIR/.venv/bin/python" "$LEAFCHECK_DIR/../.venv/bin/python" "$(command -v python3 || true)"; do
  if [[ -x "$candidate" ]] && "$candidate" -c 'import importlib.util; raise SystemExit(0 if all(importlib.util.find_spec(name) for name in ("torch", "PIL")) else 1)' 2>/dev/null; then
    LEAFCHECK_PYTHON="$candidate"
    break
  fi
done

if [[ -z "$LEAFCHECK_PYTHON" ]]; then
  print "LeafCheck needs Python with PyTorch and Pillow."
  print "See the One-time setup instructions in README.md, then double-click this file again."
  if [[ -t 0 ]]; then read -r "reply?Press Return to close. "; fi
  exit 1
fi

cd "$LEAFCHECK_DIR"
print "Starting LeafCheck on your Mac."
print "All photos and predictions stay local."
browser_args=()
if [[ "${LEAFCHECK_OPEN_BROWSER:-1}" != "0" ]]; then browser_args=(--open-browser); fi
exec "$LEAFCHECK_PYTHON" "$LEAFCHECK_DIR/backend/server.py" "${browser_args[@]}" "$@"
