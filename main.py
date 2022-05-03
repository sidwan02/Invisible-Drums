import numpy as np
from aggregator import aggregate_centroids, get_confidence_score
from detector import *

def run_single_iteration(cluster_centroids, iter_num):
    """
    runs our system for a single iteration

    Args:
        clusters: list of (row, col, intensity) tuples, one for each of the k cluster centroids
        iter_num: current iteration number (zero-indexed) for a video 
    Returns
        Nothing
    """
    print("== ITERATION ", iter_num, " ==")

    # convert to np arrays and unpack
    cluster_centroids = np.array(cluster_centroids) # list of (r,c,intensity)
    centroid_locations = cluster_centroids[:, :2] # first two coordinates are (r,c)
    centroid_intensities = cluster_centroids[:, 2] # last coordinate is intensity

    # step 3: get blobs (aggregated clusters) and calculate the movement confidence score for this frame
    blob_locs, blob_intensities = aggregate_centroids(centroid_locations, centroid_intensities)
    curr_confidence_score = get_confidence_score(blob_locs, blob_intensities, total_num_clusters=50)  # returns scalar

    # update state of prev scores
    n_prev_scores.append(curr_confidence_score)
    m_prev_scores.append(curr_confidence_score)

    # break and reiterate until we have at least max(m,n) scores
    if len(n_prev_scores) < n+1 or len(m_prev_scores) < m+1: # add 1 because we pass in prev_scores[:-1] to rebound detectors
        return
    
    # keep length at most n or m previous scores by removing oldest entries (at start of arr)
    if len(n_prev_scores) > n+1:  
        n_prev_scores.pop(0)
    if len(m_prev_scores) > m+1:
        m_prev_scores.pop(0)

    # step 4: detect potential rebound
    (
        curr_frame_has_potential_rebound,
        highest_intensity_pixel_loc, # may be used later 
    ) = detect_potential_rebound(curr_confidence_score, np.array(n_prev_scores[:-1]), blob_locs, blob_intensities, threshold=T)
    # update state of prev rebounds (boolean list)
    prev_potential_rebounds.append(curr_frame_has_potential_rebound)
    if len(prev_potential_rebounds) == 2*n + 1: # keep length at most 2n
        prev_potential_rebounds.pop(0)

    # step 5: detect rebound
    rebound_location = detect_rebound(
        curr_confidence_score, np.array(m_prev_scores[:-1]), prev_potential_rebounds, blob_locs, blob_intensities, threshold=T
    )
    if rebound_location is not None:
        # step 6: get sound effect ID based on rebound location and play the sound
        drum_sound_id = get_drum_id(img_size_x, img_size_y, rebound_location)
        print("drum id: ", drum_sound_id)
        # TODO: somehow sync drum sound with video in a mp4 or live??
        # play_sound(drum_sound_id)

        # (step 7): if we have a rebound, prevent a rebound in the next 5-10 frames?


import sys

sys.path.append("raft/")
from centroids import get_centroids

if __name__ == "__main__":
    # set parameters
    n = 3
    m = 3
    T = 50
    

    # step 1: generate image flows from RAFT
    # See README.md: Generate Image Flows

    # step 2: cluster image flows by intensity
    all_centroid_data = get_centroids()
    printf(f"all_centroid_data: {all_centroid_data}")
    # dim 1 -> folder
    # dim 2 -> video
    # dim 3 -> frame
    # dim 4 -> [r, c, intensity]

    # Note: x = c, y = r

    # intensity:
    # 0 => 0 velocity
    # 255 => highest velocity

    for folder in all_centroid_data:
        for video in folder:
            # TODO: set img size of this video (for get_drum_id())
            img_size_x = ___
            img_size_y = ___

            # keep track of state across frames (modified in run_single_iteration)
            n_prev_scores = []  # scalar list
            m_prev_scores = []  # scalar list
            prev_potential_rebounds = []  # boolean list
            # get clusters for each RAFT frame for the current video
            for iter_num, frame_clusters in enumerate(video):
                # frame_clusters is of type [(r,c,intensity), ...]
                run_single_iteration(frame_clusters, iter_num)
            
            
