modify src/config.py if necessary

# NON-LIVE

<!-- To create the image flows -->

python run_inference.py

python centroids.py

# LIVE

python live_frames.py --model raft-things.pth
