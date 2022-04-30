import sys
from centroids.py import get_centroids

sys.path.append("raft/")

cluster_centers_folders = get_centroids()
# dim 1 -> folder
# dim 2 -> video
# dim 3 -> frame
# dim 4, 5 -> (r, c, intensity)

# intensity:
# 0 => 0 velocity
# 255 => highest velocity
