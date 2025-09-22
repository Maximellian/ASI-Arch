#!/bin/bash

# batch_train_all.sh
# Run all ASI-Arch experiments sequentially using train.sh, accumulating metrics

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="pipeline/files/analysis"
LOG_DIR="pipeline/files/debug"
BATCH_LOG="$LOG_DIR/batch_training.log"

# Ensure directories exist
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Initialize batch log
echo "[$(date)] Starting batch training of all experiments" | tee "$BATCH_LOG"

# Discover experiments
echo "Discovering available experiments..." | tee -a "$BATCH_LOG"
EXPERIMENTS=(
    "delta_net_hhmr"
    "deltanet_baseline"
    "deltanet_mamba"
    "deltanet_attention"
    "deltanet_conv"
    "deltanet_hybrid"
)
echo "Found ${#EXPERIMENTS[@]} experiments to run:" | tee -a "$BATCH_LOG"
printf '%s\n' "${EXPERIMENTS[@]}" | tee -a "$BATCH_LOG"

# Initialize summary CSV
SUMMARY_FILE="$RESULTS_DIR/batch_summary.csv"
echo "experiment_name,status,duration_seconds,model_params,total_benchmarks,avg_forward_time_ms,avg_throughput,completion_time" > "$SUMMARY_FILE"
echo "Initialized batch summary at $SUMMARY_FILE" | tee -a "$BATCH_LOG"

SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_EXPERIMENTS=()

BATCH_START_TIME=$(date +%s)

for i in "${!EXPERIMENTS[@]}"; do
  EXPERIMENT="${EXPERIMENTS[$i]}"
  CURRENT=$((i + 1))
  TOTAL=${#EXPERIMENTS[@]}

  echo "" | tee -a "$BATCH_LOG"
  echo "[$CURRENT/$TOTAL] Running: $EXPERIMENT" | tee -a "$BATCH_LOG"
  echo "----------------------------------------" | tee -a "$BATCH_LOG"
  EXP_START=$(date +%s)

  # Invoke the correct train.sh
  if bash train.sh "$EXPERIMENT" 2>&1 | tee -a "$BATCH_LOG"; then
    STATUS="SUCCESS"
    SUCCESS_COUNT=$((SUCCESS_COUNT+1))
  else
    STATUS="FAIL"
    FAILURE_COUNT=$((FAILURE_COUNT+1))
    FAILED_EXPERIMENTS+=("$EXPERIMENT")
  fi

  EXP_END=$(date +%s)
  DURATION=$((EXP_END - EXP_START))
  COMPLETION_TIME="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

  # Extract metrics for summary
  LAST_LINE=$(grep "^$EXPERIMENT," "$RESULTS_DIR/benchmark.csv" | tail -n1)
  MODEL_PARAMS=$(echo "$LAST_LINE" | cut -d',' -f2)
  TOTAL_BENCH=$(grep -c "^$EXPERIMENT," "$RESULTS_DIR/benchmark.csv")
  AVG_TIME=$(grep "^$EXPERIMENT," "$RESULTS_DIR/benchmark.csv" | cut -d',' -f3 | awk '{sum+=$1} END{print NR?sum/NR:0}')
  AVG_TP=$(grep "^$EXPERIMENT," "$RESULTS_DIR/benchmark.csv" | cut -d',' -f5 | awk '{sum+=$1} END{print NR?sum/NR:0}')

  echo "$EXPERIMENT,$STATUS,$DURATION,$MODEL_PARAMS,$TOTAL_BENCH,$AVG_TIME,$AVG_TP,$COMPLETION_TIME" \
    >> "$SUMMARY_FILE"
  
  PROGRESS=$((CURRENT*100/TOTAL))
  ELAPSED=$(( $(date +%s) - BATCH_START_TIME ))
  ETA=$(( (ELAPSED*TOTAL/CURRENT) - ELAPSED ))
  echo "Progress: $PROGRESS% ($CURRENT/$TOTAL) | Elapsed: ${ELAPSED}s | ETA: ${ETA}s" | tee -a "$BATCH_LOG"
done

BATCH_END_TIME=$(date +%s)
TOTAL_TIME=$((BATCH_END_TIME - BATCH_START_TIME))

echo "" | tee -a "$BATCH_LOG"
echo "========================================" | tee -a "$BATCH_LOG"
echo "Batch complete: $SUCCESS_COUNT succeeded, $FAILURE_COUNT failed in ${TOTAL_TIME}s" | tee -a "$BATCH_LOG"
if [ $FAILURE_COUNT -gt 0 ]; then
  echo "Failed experiments:" | tee -a "$BATCH_LOG"
  printf '%s\n' "${FAILED_EXPERIMENTS[@]}" | tee -a "$BATCH_LOG"
fi
echo "Generated files:" | tee -a "$BATCH_LOG"
echo " - $RESULTS_DIR/loss.csv" | tee -a "$BATCH_LOG"
echo " - $RESULTS_DIR/benchmark.csv" | tee -a "$BATCH_LOG"
echo " - $SUMMARY_FILE" | tee -a "$BATCH_LOG"
echo " - $BATCH_LOG" | tee -a "$BATCH_LOG"
