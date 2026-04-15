#!/usr/bin/env bash

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
BIN_DEFAULT="$REPO_ROOT/build/examples/14_ampere_tf32_tensorop_gemm_multigpu_new/14_ampere_tf32_tensorop_gemm_multigpu_new"

BIN="${BIN:-$BIN_DEFAULT}"
START_CONFIG="${START_CONFIG:-0}"
END_CONFIG="${END_CONFIG:-26}"

if [[ ! -x "$BIN" ]]; then
  echo "Executable not found: $BIN" >&2
  echo "Build it first, or override BIN=/path/to/14_ampere_tf32_tensorop_gemm_multigpu_new" >&2
  exit 1
fi

printf "%-12s %-14s %-14s %-8s\n" "tile-config" "runtime-ms" "gflops" "status"

for config in $(seq "$START_CONFIG" "$END_CONFIG"); do
  cmd=(
    "$BIN"
    "--tile-config=$config"
    "--storage-device=0"
    "--compute-device=1"
    "$@"
  )

  if output="$("${cmd[@]}" 2>&1)"; then
    exit_code=0
  else
    exit_code=$?
  fi

  runtime_ms=$(awk '/Runtime:/ {print $2}' <<<"$output" | tail -n 1)
  gflops=$(awk '/GFLOPs:/ {print $2}' <<<"$output" | tail -n 1)

  if grep -q '^Passed$' <<<"$output"; then
    status="Passed"
  elif grep -q '^Failed$' <<<"$output"; then
    status="Failed"
  else
    status="Error($exit_code)"
  fi

  printf "%-12s %-14s %-14s %-8s\n" \
    "$config" "${runtime_ms:--}" "${gflops:--}" "$status"

  if [[ "$status" != "Passed" ]]; then
    echo "----- tile-config=$config raw output -----" >&2
    echo "$output" >&2
    echo "-----------------------------------------" >&2
  fi
done
