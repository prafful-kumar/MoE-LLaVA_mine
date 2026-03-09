#!/bin/bash

LOG_FILE="fisher_TS_norm_training_norm_input_norm_weight.log"

# Clear the log file from a previous run
> $LOG_FILE

while true; do
  echo "Attempting to run the script... Output is being saved to $LOG_FILE"
  
  # Run the script and append its output (stdout and stderr) to the log file
  # The 2>&1 ensures both are captured.
  # The >> appends the new output to the end of the file.
  if bash scripts/v1/stablelm/router_hyp_dyn.sh >> $LOG_FILE 2>&1; then
    # The script completed successfully
    echo "Script finished successfully without a CUDA out of memory error."
    break
  else
    # The script failed. Now we need to check the log file for the specific error.
    if grep -q "CUDA out of memory" $LOG_FILE; then
      echo "CUDA out of memory error detected. Retrying..."
      echo "--------------------------------------------------"
    else
      # The script failed for a different reason.
      echo "Script failed with a different error. Check $LOG_FILE for details."
      break
    fi
  fi
done

echo "Loop terminated. Exiting."