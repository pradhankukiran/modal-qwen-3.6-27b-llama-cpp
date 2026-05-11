#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL to your deployed Modal endpoint URL}"
: "${OPENAI_API_KEY:=local-test-key}"
: "${OPENAI_MODEL:=qwen3.6-27b}"

curl -L -sS "${OPENAI_BASE_URL%/}/warmup" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  --data "$(jq -n --arg model "$OPENAI_MODEL" '{model: $model}')"

printf '\n'
