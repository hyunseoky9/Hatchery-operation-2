#!/bin/bash
# ---------------------------------------------
# Usage: ./run_td3.sh <paramID>
# Example: ./run_td3.sh 57
# ---------------------------------------------

PARAM_ID=$1

# Generate a random session ID between 0 and 100000
SESSION_ID=$(( RANDOM % 100001 ))
SESSION_NAME="${SESSION_ID}_${PARAM_ID}_TD3"
LOGFILE="train_${PARAM_ID}_${SESSION_ID}.log"

echo "Launching TD3 training:"
echo "  paramID: ${PARAM_ID}"
echo "  tmux session: ${SESSION_NAME}"
echo "  log file: ${LOGFILE}"

# Start tmux session
tmux new-session -d -s "$SESSION_NAME" "source ~/miniconda3/bin/activate hatchery && cd ~/Hatchery-operation-2 && python TD3_env3_x_main_HPC.py ${PARAM_ID} | tee -a ${LOGFILE}"

echo "Training started in tmux session '${SESSION_NAME}'."
echo "To attach:   tmux attach -t ${SESSION_NAME}"
echo "To view log: tail -f ${LOGFILE}"
