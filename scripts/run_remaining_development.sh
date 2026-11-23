#!/bin/zsh
# Start this from the project root or through the documented detached session.
set -eu
cd "${0:A:h}/.."

run_dir=runs/final177
case "${1:-}" in
  --locked)
    .venv/bin/python scripts/run_native_resolution_check.py >> "$run_dir/native_resolution_driver.log" 2>&1
    .venv/bin/python scripts/run_final_development.py >> "$run_dir/final_development_driver.log" 2>&1
    ;;
  '')
    mkdir -p "$run_dir"
    exec /usr/bin/caffeinate -i /usr/bin/lockf -k -t 0 "$run_dir/native_resolution.lock" /bin/zsh "$0" --locked
    ;;
  *)
    print -u2 'Usage: zsh scripts/run_remaining_development.sh'
    exit 2
    ;;
esac
