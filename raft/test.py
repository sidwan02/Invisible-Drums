import glob
import os

path = "../data/DAVIS2016/JPEGImages/480p/*/*.jpg"
print(f"path: {path}")
print(glob.glob(path, recursive=True))
