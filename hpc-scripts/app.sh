#!/bin/bash

#SBATCH --job-name=QuranDatasetApp
#SBATCH --account=shams035
#SBATCH --output=app.out
#SBATCH --time=3-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24


echo 'Starting QuranDataset App #########'
bash ~/.bashrc
source ~/data/miniconda3/bin/activate
conda activate prepare-ds
cd ../frontend/
LOG_DIR=../hpc-scripts

# Run Streamlit, and direct both stdout and stderr to streamlit.log
python -m streamlit run streamlit_app.py --server.port 8080 > $LOG_DIR/streamlit.log 2>&1 &

# Capture the PID of Streamlit so we can monitor it
STREAMLIT_PID=$!
echo "streamlitPID=$STREAMLIT_PID"

# Start ngrok, and direct both stdout and stderr to ngrok.log
cd $LOG_DIR
ngrok http 8080 > ngrok.log 2>&1 &

# Capture the PID of ngrok so we can monitor it
NGROK_PID=$!
echo "ngrokPID=$NGROK_PID"


# Wait for ngrok to start (give it a few seconds)
sleep 10

# Use ngrok's local API to get the public URL and save it to a file
NGROK_URL=$(curl -s -X GET 'localhost:4040/api/tunnels' |  grep -o '"public_url":"[^"]*' | sed 's/"public_url":"//')
echo $NGROK_URL > ngrok_url.txt
echo "ngrok url: $NGROK_URL"

# Wait for both processes to finish
wait $STREAMLIT_PID
wait $NGROK_PID
