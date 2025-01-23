import numpy as np

# check if the input is a numpy array. Make sure the array is 2-dimensional
def check_type(X):
    if type(X) != np.ndarray:
        raise TypeError("The input must be a numpy array.")
    if len(X.shape) == 1:
        return X.reshape(-1, 1)
    return X

"""
The selection and the naming convension of the classes, methods, parameters and attributes follow these websites:
linear_model.LinearRegression: https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html
linear_model.Ridge: https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html
preprocessing.PolynomialFeatures: https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
model_selection.train_test_splits: https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.train_test_split.html
model_selection.cross_val_score: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
"""
class linear_model: 
    class LinearRegression: 
        def __init__(self, fit_intercept=True):
            self.coef_ = None
            self.intercept_ = None
            self.w = None
            self.fit_intercept = fit_intercept
            self.X = None
            self.y = None

        def fit(self, X, y):
            self.X = X #assign X and y to the class attributes, so that X and y won't be mutated later.
            self.y = y
            self.X = check_type(self.X)
            self.y = check_type(self.y)
            if self.fit_intercept:
                self.X = np.hstack([np.ones((self.X.shape[0], 1)), self.X]) #Add a column of 1s to X
            self.w = np.dot(np.linalg.pinv(self.X), self.y) #Conduct MLE for w
            if self.fit_intercept:
                self.coef_ = self.w[1:]
                self.intercept_ = self.w[0]               
            else:
                self.coef_ = self.w
            return self

        def predict(self, X):
            self.X = X
            self.X = check_type(self.X)
            if self.fit_intercept:  
                self.X = np.hstack([np.ones((self.X.shape[0], 1)), self.X])
            return np.dot(self.X, self.w)
    
        def score(self, X, y):
            self.X = X
            self.y = y
            self.X = check_type(self.X)
            self.y = check_type(self.y)
            residual_ss = np.sum((self.predict(self.X) - self.y)**2) #Calculate the residual sum of squares
            total_ss = np.sum((self.y - np.mean(self.y))**2) #Calculate the total sum of squares
            return 1 - residual_ss/total_ss #Calculate the R^2 score
    
    class Ridge:
        def __init__(self, alpha=1.0, fit_intercept=True):
            self.coef_ = None
            self.intercept_ = None
            self.w = None
            self.fit_intercept = fit_intercept
            if alpha < 0:
                raise ValueError("alpha must be non-negative.")
            self.alpha = alpha
            self.X = None
            self.y = None

        def fit(self, X, y):
            if self.alpha == 0: # If alpha = 0, then ridge regression is equivalent to OLS.
                linreg = linear_model.LinearRegression(fit_intercept=self.fit_intercept)
                linreg.fit(X, y)
            else:
                self.X = X
                self.y = y
                self.X = check_type(self.X)
                self.y = check_type(self.y)
                self.y.reshape(-1,)
                if self.fit_intercept: 
                    self.X = np.hstack([np.ones((self.X.shape[0], 1)), self.X])
                    self.w = np.dot(np.dot(np.linalg.inv( np.dot(np.transpose(self.X),self.X)+self.alpha*np.identity(self.X.shape[1])),
                                            np.transpose(self.X)), self.y) #Conduct MLE for w
                
            if self.fit_intercept:
                self.coef_ = self.w[1:]
                self.intercept_ = self.w[0]   
            else:
                self.coef_ = self.w
            return self
        
        def predict(self, X):
            self.X = X  
            self.X = check_type(self.X)
            if self.fit_intercept:  
                self.X = np.hstack([np.ones((self.X.shape[0], 1)), self.X])
            return np.dot(self.X, self.w)
    
        def score(self, X, y):
            self.X = X
            self.y = y
            self.X = check_type(self.X)
            self.y = check_type(self.y)
            residual_ss = np.sum((self.predict(X) - self.y)**2)
            total_ss = np.sum((self.y - np.mean(self.y))**2)
            return 1 - residual_ss/total_ss
            
            

# class for polynomial transformation
class preprocessing:
    class PolynomialFeatures:
        def __init__(self, degree):
            self.degree = degree
        
        def fit_transform(self, X, y=None): 
            self.X = X
            self.X = check_type(self.X)
            
            return np.hstack([self.X ** i for i in range(1, self.degree + 1)])  
                #Doesn't include the 0th power of X, as this will be done in the fitting process.

class model_selection:
    def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True):
        if test_size==None:
            if train_size==None:
                test_size = 0.25 
            elif 0<train_size and train_size<1:
                test_size = 1 - train_size
            else:
                raise ValueError("The range of train_size is (0,1).")
        else:
            if 0<test_size and test_size<1:
                pass
            else:
                raise ValueError("The range of test_size is (0,1).")
        
        np.random.seed(random_state) #Allows np.random to reproduce the same result for the same random_state
        data1 = arrays[0]
        data1 = check_type(data1) 
        test_data_len = round(len(data1) * test_size) # test data size

        if len(arrays)==1:
            if shuffle == True:
                np.random.shuffle(data1)
            data1_test = data1[:test_data_len]
            data1_train = data1[test_data_len:]
            return data1_train, data1_test
        else:
            data2 = arrays[1]
            data2 = check_type(data2)
            combined_data = np.hstack([data1, data2]) #Combine data1 and data2 to shuffle them in a synchronised fashion.
            if shuffle == True:
                np.random.shuffle(combined_data)
            shuffled_data1 = combined_data[:,:-1*data2.shape[1]] #Separate the data again
            shuffled_data2 = combined_data[:,-1*data2.shape[1]].reshape(-1,1)  #Separate the data again
            data1_test = shuffled_data1[:test_data_len]
            data1_train = shuffled_data1[test_data_len:]
            data2_test = shuffled_data2[:test_data_len]
            data2_train = shuffled_data2[test_data_len:]
            return data1_train, data1_test, data2_train, data2_test
  
    def cross_val_score(estimator, X, y, cv=None):
        if cv == None:
            cv = 5

        if len(X)<cv:
            raise Exception("The data size must be larger than cv.")
        
        X = check_type(X)
        y = check_type(y)

        scores_list = [] #The list to store the scores for each train-test pair.
        for i in range(cv):
            start_index = round(len(X) * i / cv)
            end_index = round(len(X) * (i + 1) / cv)
            X_test = X[start_index:end_index]
            X_train = np.vstack((X[:start_index], X[end_index:]))
            y_test = y[start_index:end_index]
            y_train = np.vstack((y[:start_index], y[end_index:]))
            score = float(estimator.fit(X_train, y_train).score(X_test, y_test))
            scores_list.append(score)
            
        return scores_list
                
            