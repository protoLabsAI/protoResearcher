#!/usr/bin/env bash
# run_eval.sh — End-to-end evaluation of nanobot vs LangGraph backends.
#
# Usage:
#   ./evals/run_eval.sh              # Full eval (calls LLM)
#   ./evals/run_eval.sh --dry-run    # Validate harness only
#
# This script:
#   1. Runs the eval suite against the nanobot backend
#   2. Runs the eval suite against the langgraph backend
#   3. Compares both result files and prints a markdown report

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
EXTRA_ARGS="${*}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo " protoResearcher Evaluation Suite"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo ""

# --- Phase 1: Nanobot backend ---
echo ">>> Phase 1: Running eval with AGENT_BACKEND=nanobot"
echo ""

cd "$PROJECT_DIR" && AGENT_BACKEND=nanobot $PYTHON -m evals.runner --backend nanobot $EXTRA_ARGS

# Find the most recent nanobot result file
NANOBOT_RESULT=$(ls -t "$RESULTS_DIR"/nanobot_*.json 2>/dev/null | head -1)
if [ -z "$NANOBOT_RESULT" ]; then
    echo "ERROR: No nanobot result file found."
    exit 1
fi
echo "Nanobot results: $NANOBOT_RESULT"
echo ""

# --- Phase 2: LangGraph backend ---
echo ">>> Phase 2: Running eval with AGENT_BACKEND=langgraph"
echo ""

cd "$PROJECT_DIR" && AGENT_BACKEND=langgraph $PYTHON -m evals.runner --backend langgraph $EXTRA_ARGS

# Find the most recent langgraph result file
LANGGRAPH_RESULT=$(ls -t "$RESULTS_DIR"/langgraph_*.json 2>/dev/null | head -1)
if [ -z "$LANGGRAPH_RESULT" ]; then
    echo "ERROR: No langgraph result file found."
    exit 1
fi
echo "LangGraph results: $LANGGRAPH_RESULT"
echo ""

# --- Phase 3: Compare ---
echo ">>> Phase 3: Generating comparison report"
echo ""

REPORT_FILE="$RESULTS_DIR/comparison_$(date -u '+%Y%m%d_%H%M%S').md"
cd "$PROJECT_DIR" && $PYTHON -m evals.compare "$NANOBOT_RESULT" "$LANGGRAPH_RESULT" --output "$REPORT_FILE"

echo ""
echo "============================================================"
echo " Comparison report: $REPORT_FILE"
echo "============================================================"
echo ""

# Print the report to stdout as well
cat "$REPORT_FILE"
