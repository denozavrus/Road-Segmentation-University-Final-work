import cv2
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def get_road_mask(outputs : torch.Tensor, image) -> np.array:
    mask = (outputs > 0.5).int().cpu().detach().numpy() # Get mask of outputs
    mask = mask[0][0]
    mask = 255 * mask.astype(np.uint8)

    im, contours, hierarchy=cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Draw contours on image
    image_copy = image.copy
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    shapes = [con.shape[0] for con in contours] # get biggest contour
    main_road_index = shapes.index(max(shapes))
    contours[main_road_index] = contours[main_road_index].reshape((contours[main_road_index].shape[0], 2))
    return contours[main_road_index]

def fit_polynom(contours, degree = 3):
    # Function to create a poly Linear Regression
    def PolynomialRegression(degree) -> Pipeline:
        return Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('std_scaler', StandardScaler()),
            ('lin_reg', LinearRegression())
        ])
    clf = PolynomialRegression(degree)
    X = np.array(contours[:, 0]).reshape(-1, 1)
    Y = np.array(contours[:, 1]).reshape(-1, 1)
    clf.fit(X, Y)
    return clf