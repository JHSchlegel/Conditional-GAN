#!/bin/bash

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate cgan

# Start TensorBoard in the background and save its process ID (PID)
tensorboard --logdir=logs --port=6006 &
TENSORBOARD_PID=$!

xdg-open http://localhost:6006

# Start the training script
python train.py


# Wait for 5 seconds after train.py finishes
sleep 5

# Start the evaluation script
python validation.py

# Sleep for 10 seconds before killing the TensorBoard process
sleep 10

kill $TENSORBOARD_PID

# Wait a bit for TensorBoard to release the port
sleep 5