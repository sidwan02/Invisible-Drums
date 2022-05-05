# Invisible-Drums

## Requirements:

In your desired environment:

`pip install -r requirements.txt`

## Generate Image Flows

### From Folder

```
cd raft/
python run_inference.py --clear
```

### Live Webcam

`python live_frames.py --model <path/to/raft/model>`

Example:

`python live_frames.py --model raft-things.pth`

## Get Maximal Intensity Centroids

```
python centroids.py --clear
```
