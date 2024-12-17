#!/bin/bash

# Set the path to your Python environment
PYTHON_PATH="/usr/bin/python3"

# Set the path to your script
SCRIPT_PATH="/path/to/your/fetch_data.py"

# Log file
LOG_FILE="/path/to/logs/data_fetch.log"

# Run the script and log output
echo "Starting data fetch at $(date)" >> "$LOG_FILE"
$PYTHON_PATH $SCRIPT_PATH >> "$LOG_FILE" 2>&1
echo "Completed data fetch at $(date)" >> "$LOG_FILE" 