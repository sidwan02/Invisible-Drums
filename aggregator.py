import numpy as np
from sklearn.cluster import MeanShift

frame = np.array([
    [20,2,2,2,2,2,2,2,2,2,2,2,2,2,23],
    [2,2,2,2,2,2,9,2,2,2,2,2,2,2,2],
    [2,2,2,2,2,9,9,9,2,2,2,2,2,2,2],
    [2,2,2,2,2,9,9,9,2,2,2,2,2,2,2],
    [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
    [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
    [2,2,2,2,2,2,9,9,9,2,2,2,2,2,2],
    [2,2,2,2,2,2,9,9,9,2,2,2,2,2,2],
    [21,2,2,2,2,2,2,2,2,2,2,2,2,2,22]
])
print(np.take(frame, ([0,0], [8, 14])))
# should expect 2-3 blobs
centroid_locations = np.array([
    [3,4],
    [3,5],
    [4,4],
    [6,6],
    [6,7],
    [7,7],
    [7,13],
] )
centroid_intensities = np.array([3,3,4,6,7,6,10])



def aggregate_centroids(centroid_locations, centroid_intensities, raft_frame):
    '''
    Given a map of all k centroids in an optical flow frame, aggregates all nearby centroids into blobs
    using their relative pixel locations.

    Args
        centroids: dict mapping from (x,y) centroid locations to the corresponding avg intensity values.
        raft_frame: current RAFT frame, np array of size [img_height, img_width]
    Returns
        blobs_locations: (x,y) blob centroid locations. Possibly fewer entries than those in the input
                Note that the number of blobs is variable depending on the relative centroid distances.
        blob_intensities: [n_blobs,] array of scalar intensity at each blob centroid.
    '''
    clustering = MeanShift().fit(centroid_locations)
    blob_locations = clustering.cluster_centers_ # [n_blobs, 2]
    # TODO: get intensity by iterating over cluster membership and getting avg cluster centroid int for each blob
    blob_intensities = []
    for blob_id in range(clustering.cluster_centers_):
        # get avg intensity for this blob
        blobs_in_cluster = ?????????? # [cluster_sz,]
        blob_intensities.append(np.mean(_))

    return blob_locations, np.array(blob_intensities)

def get_confidence_score(blobs):
    '''
    
    '''
    pass


aggregate_centroids(centroid_locations, centroid_intensities, frame)
