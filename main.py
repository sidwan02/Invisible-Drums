import numpy as np
from aggregator import aggregate_centroids, get_confidence_score
from detector import *
from argparse import ArgumentParser
from meanshift import mark_centroids
import copy
import cv2


def mark_blobs(grayscale, blob_centers, blob_intensities):
    orig = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    overlay = orig.copy()

    for i, (r, c) in enumerate(blob_centers):
        r, c = round(r), round(c)
        alpha = len(blob_intensities[i]) / 50
        overlay = cv2.circle(
            overlay, (c, r), radius=30, color=(0, 0, 255), thickness=-1
        )
        orig = cv2.addWeighted(overlay, alpha, orig, 1 - alpha, 0, orig)

    return orig


def run_single_iteration(cluster_centroids, frame_flow, blobs_path=None):
    # print(f"cluster_centroids: {cluster_centroids}")

    # convert to np arrays and unpack
    cluster_centroids = np.array(cluster_centroids)  # list of (r,c,intensity)
    centroid_locations = cluster_centroids[:, :2]  # first two coordinates are (r,c)
    centroid_intensities = cluster_centroids[:, 2]  # last coordinate is intensity

    # step 3: get blobs (aggregated clusters) and calculate the movement confidence score for this frame
    blob_locs, blob_intensities = aggregate_centroids(
        centroid_locations, centroid_intensities
    )
    curr_confidence_score = get_confidence_score(
        blob_locs, blob_intensities, total_num_clusters=50
    )  # returns scalar

    if blobs_path is not None:
        # blobs_on_flow = mark_centroids(copy.deepcopy(frame_flow), blob_locs)
        blobs_on_flow = mark_blobs(
            copy.deepcopy(frame_flow), blob_locs, blob_intensities
        )

        cv2.imwrite(blobs_path, blobs_on_flow)

    # print(f"blob_locs: {blob_locs}")
    # print(f"blob_intensities: {blob_intensities}")
    # print(f"curr_confidence_score: {curr_confidence_score}")

    return {
        "blob_locs": blob_locs,
        "blob_intensities": blob_intensities,
        "curr_confidence_score": curr_confidence_score,
    }


def rebound_detection(frame_blob_data, iter_num):
    """
    runs our system for a single iteration

    Args:
        clusters: list of (row, col, intensity) tuples, one for each of the k cluster centroids
        iter_num: current iteration number (zero-indexed) for a video 
    Returns
        Nothing
    """
    print("== VIDEO ", iter_num, " ==")

    blob_locs, blob_intensities, curr_confidence_score, = (
        frame_blob_data["blob_locs"],
        frame_blob_data["blob_intensities"],
        frame_blob_data["curr_confidence_score"],
    )

    print(f"blob_intensities: {blob_intensities}")

    # update state of prev scores
    n_prev_scores.append(curr_confidence_score)
    m_prev_scores.append(curr_confidence_score)

    # break and reiterate until we have at least max(m,n) scores
    if (
        len(n_prev_scores) < n + 1 or len(m_prev_scores) < m + 1
    ):  # add 1 because we pass in prev_scores[:-1] to rebound detectors
        return

    # keep length at most n or m previous scores by removing oldest entries (at start of arr)
    if len(n_prev_scores) > n + 1:
        n_prev_scores.pop(0)
    if len(m_prev_scores) > m + 1:
        m_prev_scores.pop(0)

    # step 4: detect potential rebound
    (
        curr_frame_has_potential_rebound,
        highest_intensity_pixel_loc,  # may be used later
    ) = detect_potential_rebound(
        curr_confidence_score,
        np.array(n_prev_scores[:-1]),
        blob_locs,
        blob_intensities,
        threshold=T,
    )
    # update state of prev rebounds (boolean list)
    prev_potential_rebounds.append(curr_frame_has_potential_rebound)
    if len(prev_potential_rebounds) == 2 * n + 1:  # keep length at most 2n
        prev_potential_rebounds.pop(0)

    # step 5: detect rebound
    rebound_location = detect_rebound(
        curr_confidence_score,
        np.array(m_prev_scores[:-1]),
        prev_potential_rebounds,
        blob_locs,
        blob_intensities,
        threshold=T,
    )

    print(f"rebound_location: {rebound_location}")

    if rebound_location is not None:
        # step 6: get sound effect ID based on rebound location and play the sound
        drum_sound_id = get_drum_id(img_size_x, img_size_y, rebound_location)
        print("drum id: ", drum_sound_id)
        # TODO: somehow sync drum sound with video in a mp4 or live??
        # play_sound(drum_sound_id)

        # (step 7): if we have a rebound, prevent a rebound in the next 5-10 frames?

    return


import sys

sys.path.append("raft/")
# from centroids import get_centroids

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--path", type=str, default="./data/custom/")
    # set parameters
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--T", type=int, default=50)

    args = parser.parse_args()

    n = args.n
    m = args.m
    T = args.T

    # step 1: generate image flows from RAFT
    # See README.md: Generate Image Flows

    # step 2: cluster image flows by intensity
    # See README.md: Get Maximal Intensity Centroids

    # all_centroid_data = get_centroids()

    all_blobs_data = np.load(args.path + "all_blobs_data.npy", allow_pickle=True)
    # all_flow_grayscale = np.load(args.path + "all_flow_grayscale.npy")

    print(f"all_blobs_data: ", all_blobs_data)

    # dim 1 -> folder/video
    # dim 2 -> frame
    # dim 3 -> {dict of three keys and values}

    # Note: x = c, y = r

    # intensity:
    # 0 => 0 velocity
    # 255 => highest velocity

    # each folder represents a video
    for video in all_blobs_data:
        # TODO: set img size of this video (for get_drum_id())
        c, r = (856, 480)
        img_size_x = c
        img_size_y = r

        # keep track of state across frames (modified in run_single_iteration)
        n_prev_scores = []  # scalar list
        m_prev_scores = []  # scalar list
        prev_potential_rebounds = []  # boolean list
        # get clusters for each RAFT frame for the current video
        for iter_num, frame_blob_data in enumerate(video):
            # frame_clusters is of type [[r,c,intensity], ...]
            rebound_detection(frame_blob_data, iter_num)
