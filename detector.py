import numpy as np
from aggregator import get_confidence_score

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
def detect_rebound(previous_scores, previous_rebounds, locations, intensities, threshold= 0.5):
    # getting the confidence score and potential_rebound_percent
    confidence_score = get_confidence_score(locations, intensities)
    potential_rebound_percent = np.count_nonzero(previous_rebounds) / np.shape(previous_rebounds)[0]
    # calculating the max intensity location
    index = 0
    intensitiy = intensities[0]
    for i in range(np.shape(intensities)[0]):
        if intensities[i] > intensitiy:
            index = i
    # sereing if the score is greater than previous m scores, m being a hyperparameter
    is_greater = True
    for score in previous_scores:
        if score > confidence_score:
            is_greater = False
    # checking if all conditions pass
    if(confidence_score > threshold):
        if(potential_rebound_percent > 0.3):
            if(is_greater):
                return locations[index]
    # returning null if not
    else:
        return None



"""
function that helps in detect of a rebound took place. 

INPUTS:
* n_scores = list of past n scores (or a smaller list if we’ve not seen n frames yet)
* pix_loc = list of pixel locations of all blob centroids for the current frame
* intensities = list of pixel locations of all blob centroids for the current frame
* threshold = threshold is the value which the current score must be under

OUTPUTS: 
* boolean for whether the current score suggests a potential incoming rebound for the current frame
* (x,y) pixel location of most intense centroid 
"""
def detect_rebound(n_scores, pix_loc, intensities, threshold):
    current_score = get_confidence_score(pix_loc, intensities)
    highest_intensity_loc = pix_loc[np.argmax(intensities)]
    return (np.all(n_scores < current_score) and current_score < threshold, highest_intensity_loc)


