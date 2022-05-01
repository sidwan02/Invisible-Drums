import numpy as np
from collections import defaultdict
from sklearn.cluster import MeanShift

def aggregate_centroids(centroid_locations, centroid_intensities):
    '''
    Given all k cluster centroids in an RAFT optical flow frame, aggregates all nearby centroids into blobs
    using their relative pixel locations and calculate the corresponding avg intensities.

    Args
        centroid_locations: (x,y) centroid locations 
        centroid_intensities: pixel values corresponding to the centroid locations.
    Returns
        blob_locations: (x,y) blob centroid locations. Possibly fewer entries than those in the input. [n_blobs, 2].
                Note that the number of blobs is variable depending on the relative centroid distances.
        blob_intensities: dictionary mapping from blob ID to a list of intensities of the clusters in that blob (e.g. {1: [200,210,190]} )
    '''
    # TODO: set bandwidth = fraction of rez?
    clustering = MeanShift(bandwidth=40).fit(centroid_locations)
    blob_locations = clustering.cluster_centers_ # [n_blobs, 2]
    # get intensity by iterating over cluster membership and getting avg cluster centroid int for each blob
    blob_intensities = defaultdict(list) # map from blob ID/label to its centroid's intensity (0-255)
    for blob_id, cluster_intensity in zip(clustering.labels_, centroid_intensities):
        blob_intensities[blob_id].append(cluster_intensity) #default 0
        
    # # iterate over map to get avg blob intensity
    # blob_avg_intensities = []
    # for blob_id in range(clustering.cluster_centers_.shape[0]):
    #     blob_avg_intensities.append(np.mean(blob_intensities[blob_id]))
    # print(np.array(blob_avg_intensities))

    print(blob_intensities)
    return blob_locations, blob_intensities

def get_confidence_score(blob_locations, blob_intensities, total_num_clusters=50):
    '''
    Calculates the movement confidence score for the current frame with the following desired qualities:
        - More blobs = lower score (since blobs are likely scattered)
        - Higher intensity = higher score (high intensity = more motion)
        - Higher distance = lower score (regardless of the number of blobs, there is uncertainty in hand movement if the blobs are very far apart)

    Args:
        blob_locations: (x,y) blob centroid locations. 
        blob_intensities: dictionary mapping from blob ID to a list of intensities of the clusters in that blob
        num_clusters: fixed number of clusters given to aggregate_centroids (50)
    Returns
        confidence score: positive scalar
    '''
    # calculate weighted avg according to num of clusters that converge at each blob
    weighted_intensities = 0
    for blob_id, cluster_intensities in blob_intensities.items():
        weighted_intensities += np.mean(cluster_intensities) * len(cluster_intensities)
    weighted_intensities /= total_num_clusters
    num_blobs = blob_locations.shape[0]
    return weighted_intensities / num_blobs


## TESTING
# frame = np.array([
#     [20,2,2,2,2,2,2,2,2,2,2,2,2,2,23],
#     [2,2,2,2,2,2,9,2,2,2,2,2,2,2,2],
#     [2,2,2,2,2,9,9,9,2,2,2,2,2,2,2],
#     [2,2,2,2,2,9,9,9,2,2,2,2,2,2,2],
#     [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
#     [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
#     [2,2,2,2,2,2,9,9,9,2,2,2,2,2,2],
#     [2,2,2,2,2,2,9,9,9,2,2,2,2,2,2],
#     [21,2,2,2,2,2,2,2,2,2,2,2,2,2,22]
# ])
# # should expect 2-3 blobs
# centroid_locations = np.array([
#     # k =7
#     [1,1],
#     [0,1],
#     [0,0],

#     [500,450],
#     [520,430],

#     [530,440],
#     [550,450],
# ] )
# centroid_intensities = np.array([200,200,200,  150,150, 150,150])

# blob_locs, blob_ints = aggregate_centroids(centroid_locations, centroid_intensities)
# print(get_confidence_score(blob_locs, blob_ints))