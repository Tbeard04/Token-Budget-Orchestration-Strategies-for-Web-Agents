# Applies the environment patches recorded in the project journal. 
# Each file is backed up once to <file>.bak so its safe to re-run.

set -euo pipefail
 
PY="${PYTHON:-python}"
SP="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
 
backup() { [ -f "$1.bak" ] || cp "$1" "$1.bak"; }
 
#1. Evaluator model: replace the retired gpt-4-1106-preview with gpt-4o-mini
for f in "$SP/webarena/llms/providers/openai_utils.py" \
         "$SP/webarena/evaluation_harness/helper_functions.py"; do
  [ -f "$f" ] || continue
  backup "$f"
  sed -i 's/gpt-4-1106-preview/gpt-4o-mini/g' "$f"
  echo "evaluator model set to gpt-4o-mini: $f"
done