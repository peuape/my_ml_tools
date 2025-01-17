'''
1. linreg attributes: coef, intercept
•fit: determines the coefficients. input: (np.array)X, (np.array)y
•predict: applies the coefficients to the input. input: (np.array)X output: (np.array)y
•score: calculates the R^2 value. input: (np.array)X, (np.array)y, metric("mse" or "mae")

2. polynomial input: (int)degree
•transform: transforms the input into a polynomial. input: (np.array)X : (np.array)X

'''

import numpy as np

# check if the input is a numpy array
def check_type(x):
    if type(x) != np.ndarray:
        raise TypeError("The input has to be a numpy array.")
    


class linear_model:
# linear regression class
    class LinearRegression:
        def __init__(self):
            self.coef_ = None
            self.intercept_ = None
            self.w = None

        def fit(self, X, y):
            check_type(X)
            check_type(y)
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            self.w = np.dot(np.linalg.pinv(X), y)
            self.coef_ = self.w[1:]
            self.intercept_ = self.w[0]               
            print(self.w)
            return self


        def predict(self, X):
            check_type(X)
            return np.dot(X, self.w)
    
        def score(self, X, y, metric="mse"):
            check_type(X)
            try:
                if metric == "mse":
                    return np.mean((self.predict(X) - y) ** 2)
                elif metric == "mae":
                    return np.mean(np.abs(self.predict(X) - y))
            except:
                print("The metric has to be either 'mse' or 'mae'. ")

# class for polynomial transformation
class preprocessing:
    class PolynomialFeatures:
        def __init__(self, degree, include_bias=True):
            self.degree = degree
            self.include_bias = include_bias

        def transform(self, X):
            check_type(X)
            if self.include_bias:
                X_poly = np.hstack([X ** i for i in range(self.degree + 1)])
                return X_poly
            else:
                return np.hstack([X ** i for i in range(1, self.degree + 1)])      

class model_selection:
    def train_test_split(self, data, test_size=None, train_size=None, random_state=None, shuffle=True):
        pass    
