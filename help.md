modify src/config.py if necessary

cd raft/

# NON-LIVE

python run_inference.py

python centroids.py

# LIVE

python live_frames.py --model raft-things.pth
