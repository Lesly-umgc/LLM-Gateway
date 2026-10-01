#!/bin/sh
# Ollama entrypoint for the compose stack.
# Starts the server, then pulls the model once (skipped if already present).
# POSIX sh only: compose runs this under /bin/sh, so no pipefail/bashisms.
set -eu

MODEL="${OLLAMA_MODEL:-llama3.2:3b}"

ollama serve &
SERVER_PID=$!

# wait for the API to come up
for _ in $(seq 1 60); do
  if ollama list >/dev/null 2>&1; then break; fi
  sleep 1
done

if ollama list 2>/dev/null | grep -q "^${MODEL%%:*}"; then
  echo "model ${MODEL} already present, skipping pull"
else
  echo "pulling model ${MODEL} (first start only)..."
  ollama pull "${MODEL}"
fi

wait "$SERVER_PID"
