import numpy as np

"""
Gets the ID of the drum, if and when a drum is hit
INPUTS
* (x, y) location of the drum hit
* Image frame at time the drum was hit
OUTPUTS:
* Drum ID
"""
def get_drum_id(frame, location):
    Image_X_dimension = np.shape(frame)[0]
    Image_Y_dimension = np.shape(frame)[1]
    X = location[0]
    Y = location[1]
    if(X < Image_X_dimension / 2 and Y > Image_Y_dimension / 2):
        return 1
    elif(X < Image_X_dimension / 2 and Y < Image_Y_dimension / 2):
        return 2
    elif(X > Image_X_dimension / 2 and Y < Image_Y_dimension / 2):
        return 3
    elif(X > Image_X_dimension / 2 and Y > Image_Y_dimension / 2):
        return 4

