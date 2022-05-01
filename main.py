import numpy as np
from aggregator import aggregate_centroids, get_confidence_score


def run_single_iteration(cluster_centroids):
    """
    runs our system for a single iteration

    Args:
        clusters: list of (row, col, intensity) tuples, one for each of the k cluster centroids
    Returns
        Nothing
    """
    # convert to np arrays and unpack
    cluster_centroids = np.array(cluster_centroids) # list of (r,c,intensity)
    centroid_locations = cluster_centroids[:, :2] # first two coordinates are (r,c)
    centroid_intensities = cluster_centroids[:, 2] # last coordinate is intensity

    # step 3: get blobs (aggregated clusters) and calculate the movement confidence score for this frame
    blob_locs, blob_intensities = aggregate_centroids(centroid_locations, centroid_intensities)
    confidence_score = get_confidence_score()  # scalar

    # update state of prev scores
    n_prev_scores.append(confidence_score)
    m_prev_scores.append(confidence_score)
    if len(n_prev_scores) == n:
        n_prev_scores.pop(0)
    if len(m_prev_scores) == m:
        m_prev_scores.pop(0)

    # step 4: detect potential rebound
    (
        curr_frame_has_potential_rebound,
        highest_intensity_pixel_loc,
    ) = detect_potential_rebound(n_prev_scores, blob_locs, blob_intensities)
    # update state of prev rebounds
    prev_potential_rebounds.append(curr_frame_has_potential_rebound)
    if len(prev_potential_rebounds) == 2 * n:
        prev_potential_rebounds.pop(0)

    # step 5: detect rebound
    rebound_location = detect_rebound(
        m_prev_scores, prev_potential_rebounds, blob_locs, blob_intensities
    )
    if rebound_location is not None:
        # step 6: get sound effect ID based on rebound location and play the sound
        sound_id = get_drum_sound(rebound_location)
        play_sound(sound_id)

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
            for frame_clusters in video:
                # frame_clusters = [(r,c,intensity), ...]
                # keep track of previous state
                n_prev_scores = []  # scalar list
                m_prev_scores = []  # scalar list
                prev_potential_rebounds = []  # boolean list
                run_single_iteration(frame_clusters)

