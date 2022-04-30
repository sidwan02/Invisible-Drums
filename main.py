import numpy as np
from aggregator import aggregate_centroids, get_confidence_score


def run_single_iteration(r, c, intensity):
    '''
    runs our system for a single iteration
    Args:
        r: ?????
        c: ?????
        intensity:  ?????
    '''
    # step 3: get blobs (aggregated clusters) and calculate the movement confidence score for this frame
    blob_locs, blob_intensities = aggregate_centroids(curr_frame_centroids, curr_frame)
    confidence_score = get_confidence_score() # scalar

    # update state of prev scores
    n_prev_scores.append(confidence_score)
    m_prev_scores.append(confidence_score)
    if len(n_prev_scores) == n:
        n_prev_scores.pop(0)
    if len(m_prev_scores) == m:
        m_prev_scores.pop(0)

    # step 4: detect potential rebound
    curr_frame_has_potential_rebound, highest_intensity_pixel_loc = detect_potential_rebound(n_prev_scores, blob_locs, blob_intensities)
    # update state of prev rebounds
    prev_potential_rebounds.append(curr_frame_has_potential_rebound)
    if len(prev_potential_rebounds) == 2*n:
        prev_potential_rebounds.pop(0)

    # step 5: detect rebound
    rebound_location = detect_rebound(m_prev_scores, prev_potential_rebounds, blob_locs, blob_intensities)
    if rebound_location is not None:
        # step 6: get sound effect ID based on rebound location and play the sound
        sound_id = get_drum_sound(rebound_location)
        play_sound(sound_id)

        # (step 7): if we have a rebound, prevent a rebound in the next 5-10 frames?


if __name__ == "__main__":
    # set parameters
    n = 3
    m = 3
    T = 50

    # step 1/2: feed in frames to RAFT and cluster by intensity
    all_centroid_data = ___()

    for folder in all_centroid_data:
        for video in folder:
            for frame in video:
                #TODO: pls rename or document r, c, intensity
                for r, c, intensity in frame:
                    # keep track of previous state
                    n_prev_scores = [] # scalar list
                    m_prev_scores = [] # scalar list
                    prev_potential_rebounds = [] # boolean list
                    run_single_iteration(r, c, intensity)