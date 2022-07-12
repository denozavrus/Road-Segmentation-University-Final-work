from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np


def fit_polynom(contours, degree=3):

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

