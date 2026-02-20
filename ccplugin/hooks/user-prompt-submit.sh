#!/usr/bin/env bash
# UserPromptSubmit hook: optionally auto-injects relevant memory context.
# Set MEMSEARCH_AUTO_INJECT=true in settings.local.json "env" to enable.
# When disabled (default), emits a hint so the agent can use the memory-recall skill.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Skip short prompts (greetings, single words, etc.)
PROMPT=$(_json_val "$INPUT" "prompt" "")
if [ -z "$PROMPT" ] || [ "${#PROMPT}" -lt 10 ]; then
  echo '{}'
  exit 0
fi

# Need memsearch available
if [ -z "$MEMSEARCH_CMD" ]; then
  echo '{}'
  exit 0
fi

# Auto-inject mode: search and include top results as context
AUTO_INJECT="${MEMSEARCH_AUTO_INJECT:-false}"
TOP_K="${MEMSEARCH_AUTO_INJECT_TOP_K:-3}"

if [ "$AUTO_INJECT" = "true" ]; then
  # Run search with the user's prompt, capture results
  RESULTS=$($MEMSEARCH_CMD search -k "$TOP_K" "$PROMPT" 2>/dev/null || true)

  if [ -n "$RESULTS" ] && [ "$RESULTS" != "No results found." ]; then
    CONTEXT="# Memory Context (auto-injected)\n\n${RESULTS}"
    json_context=$(_json_encode_str "$CONTEXT")
    echo "{\"systemMessage\": \"[memsearch] Auto-injected ${TOP_K} memory results\", \"hookSpecificOutput\": {\"hookEventName\": \"UserPromptSubmit\", \"additionalContext\": $json_context}}"
  else
    echo '{"systemMessage": "[memsearch] Memory available (no relevant results for this prompt)"}'
  fi
else
  echo '{"systemMessage": "[memsearch] Memory available"}'
fi
