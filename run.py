import sys

sys.path.append("raft/")
from centroids import get_centroids

cluster_centers_folders = get_centroids()
print(f"cluster_centers_folders: {cluster_centers_folders}")
# dim 1 -> folder
# dim 2 -> video
# dim 3 -> frame
# dim 4 -> [r, c, intensity]

# Note: x = c, y = r

# intensity:
# 0 => 0 velocity
# 255 => highest velocity
