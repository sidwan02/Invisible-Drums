import numpy as np
from pygame import mixer

"""
function that helps in detect of a (POTENTIAL) rebound took place. 

INPUTS:
* n_scores = list of past n scores (or a smaller list if we’ve not seen n frames yet)
* pix_loc = list of pixel locations of all blob centroids for the current frame
* intensities = list of pixel locations of all blob centroids for the current frame
* threshold = threshold is the value which the current score must be under

OUTPUTS: 
* boolean for whether the current score suggests a potential incoming rebound for the current frame
* (x,y) pixel location of most intense centroid 
"""
def detect_potential_rebound(curr_confidence_score, n_prev_scores, centroid_pix_locs, intensities, threshold):
    highest_intensity_loc = centroid_pix_locs[np.argmax(intensities)]
    return (np.all(curr_confidence_score < n_prev_scores) and curr_confidence_score < threshold), highest_intensity_loc


"""
function that helps in detect of a rebound took place. 

INPUTS:
* list of m previous scores (or a smaller list if we've not seen m frames yet)
* list of 2n booleans for whether each previous frame had a potential incoming rebound
* list of pixel locations of all blob centroids for the current frame
* list of intensities of all blob centroids for the current frame
* hyperparameter threshold

OUTPUTS:
If we have a rebound then it returns the (x, y) pixel location of rebound. If there is no rebound then 
we just return None
"""
def detect_rebound(curr_confidence_score, previous_scores, previous_rebounds, locations, intensities, current_scores, idx, threshold=17):
    # getting potential_rebound_percent
    potential_rebound_percent = np.count_nonzero(previous_rebounds) / np.shape(previous_scores)[0]

    # check if the current score is greater than all m previous scores, m=hyperparameter
    curr_score_above_m_prev_scores = previous_scores[curr_confidence_score <= previous_scores].size == previous_scores.size
    check_window_smallest = False
    window = 14
    window_start = idx
    window_end = idx
    if window_start - window < 0:
        window_start = 0
    else:
        window_start = idx - window
    if window_end  + window >= len(current_scores):
        window_end = len(current_scores) - 1
    else:
        window_end = idx + window
    window_scores = np.array(current_scores[window_start:window_end])
    curr_score_window_check= np.all(curr_confidence_score <= window_scores)
    # checking if all conditions for a rebound pass
    # print("curr confidence above: ", curr_confidence_score > threshold)
    # print("<30% pot reb: ", potential_rebound_percent > 0.3)
    # print("curr confidence above: ", curr_confidence_score > threshold)
    if (curr_confidence_score < threshold) and (potential_rebound_percent > 0.12) and (curr_score_above_m_prev_scores and curr_score_window_check):
        max_intensity_location = locations[np.argmax(intensities)]
        return max_intensity_location

    # returning null if not
    return None



"""
Gets the ID of the drum, if and when a drum is hit
INPUTS
* (x, y) location of the drum hit
* Image frame at time the drum was hit
OUTPUTS:
* Drum ID
"""
def get_drum_id(img_size_x, img_size_y, location):
    X = location[0]
    Y = location[1]
    if(X < 150 and Y < 300):
        return "/Users/anishansupradhan/Desktop/CS1430/Invisible-Drums/1.mp3"
    elif(X > 400 and Y < 300):
        return "/Users/anishansupradhan/Desktop/CS1430/Invisible-Drums/2.mp3"
    elif(X < 100 and Y > 475):
        return "/Users/anishansupradhan/Desktop/CS1430/Invisible-Drums/3.mp3"
    elif(X > 400 and Y > 475):
        return "/Users/anishansupradhan/Desktop/CS1430/Invisible-Drums/4.mp3"

if __name__ == "__main__":

    ## TESTING

    curr_confidence_score = np.array(40)
    n_prev_scores= np.array([50,40,60]) 
    curr_blob_centroid_pix_locs = np.array([
        [20,10],
        [100,90],
        [500,450],
    ])
    intensities = [4, 90, 200] 
    threshold = 35



    curr_frame_has_potential_rebound, highest_intensity_pixel_loc = detect_potential_rebound(curr_confidence_score, n_prev_scores, curr_blob_centroid_pix_locs, intensities, threshold)
    print("potential rebound: ", curr_frame_has_potential_rebound)

    prev_potential_rebounds = [False, True, True, False, True, False]
    m_prev_scores= np.array([30,20,30,10]) 
    rebound_location = detect_rebound(
        curr_confidence_score, m_prev_scores, prev_potential_rebounds, curr_blob_centroid_pix_locs, intensities, threshold
    )
    print("rebound loc: ", rebound_location)