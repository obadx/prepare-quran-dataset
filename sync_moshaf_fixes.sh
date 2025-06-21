#!/bin/bash

# Configuration
REMOTE_HOST="alex"
LOCAL_DIR="../moshaf-fixes"
REMOTE_DIR="/cluster/users/shams035u1/data/mualem-recitations-annotated"
RETRY_DELAY=30  # Seconds between retries after failure

while true; do
  echo "[$(date)] Starting sync: $LOCAL_DIR -> $REMOTE_HOST:$REMOTE_DIR"
  
  # Perform sync with robust error handling
  rsync -avz --partial --progress --delete -e ssh "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR"
  
  EXIT_CODE=$?
  echo "[$(date)] Sync exited with status $EXIT_CODE"
  
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Sync completed successfully. Restarting immediately..."
  else
    echo "Sync failed! Retrying in $RETRY_DELAY seconds..."
    sleep $RETRY_DELAY
  fi
done
