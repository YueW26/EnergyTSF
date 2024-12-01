#!/bin/bash

# Parameters to vary
window_sizes=(12 24)           # Values for window_size
devices=("cuda:0")             # Values for device
horizon=(3 9 15)               # Values for horizon
test_length=(1 6 12)           # Values for test_length

# Other fixed parameters
dataset="France_processed_0"
norm_method="z_score"
train_length=7
valid_length=2
root_path="/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/"
data="Opennem"
task_name="forecasting"
data_path="France_processed_0.csv"
target="Fossil Gas  - Actual Aggregated [MW]"

# Log file to store errors
log_file="run_commands.log"

# Function to execute command
run_command() {
    local window_size=$1
    local test_l=$2
    local hor=$3
    local device=$4

    # Construct the command
    local cmd="python Stemgnn/stem_gnn.py \
        --train True \
        --evaluate True \
        --dataset $dataset \
        --window_size $window_size \
        --horizon $hor \
        --norm_method $norm_method \
        --train_length $train_length \
        --valid_length $valid_length \
        --test_length $test_l \
        --root_path $root_path \
        --device $device \
        --data $data \
        --task_name $task_name \
        --data_path $data_path \
        --target \"$target\""

    echo "Running command: $cmd"

    # Execute the command and log errors
    eval $cmd >> $log_file 2>&1
    if [ $? -ne 0 ]; then
        echo "Error occurred while executing: $cmd" | tee -a $log_file
    fi
}

# Iterate over parameter combinations
for window_size in "${window_sizes[@]}"; do
    for test_l in "${test_length[@]}"; do
        for hor in "${horizon[@]}"; do
            for device in "${devices[@]}"; do
                # Run each command in the background and limit concurrency
                run_command "$window_size" "$test_l" "$hor" "$device" &
                
                # Limit concurrency to 4 jobs at a time
                while [ $(jobs | wc -l) -ge 4 ]; do
                    sleep 1
                done
            done
        done
    done
done

# Wait for all background processes to finish
wait

echo "All commands have finished executing. Logs are saved in $log_file."

