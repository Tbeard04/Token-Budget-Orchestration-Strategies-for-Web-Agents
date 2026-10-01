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


#2. HTMLContentEvaluator: accept dict-format required_contents
f="$HOME/webarena/evaluation_harness/evaluators.py"
if [ -f "$f" ]; then
  backup "$f"
  $PY - "$f" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'clean(x) for x in required_contents.split(" |OR| ")'
new = ('clean(x) for x in (required_contents if isinstance(required_contents, str) '
       'else " |OR| ".join(map(str, required_contents.get("must_include", '
       '[required_contents.get("exact_match", "")])))).split(" |OR| ")')
if old in s and new not in s:
    open(p, "w").write(s.replace(old, new))
print(f"evaluator accepts dict required_contents: {p}")
EOF
fi


#3. BrowserGym action timeout: raise from 500 ms to 10000 ms
f="$SP/browsergym/core/action/functions.py"
if [ -f "$f" ]; then
  backup "$f"
  sed -i 's/timeout=500)/timeout=10000)/g' "$f"
  echo "action timeout set to 10000 ms ($(grep -c 'timeout=10000)' "$f") occurrences): $f"
fi