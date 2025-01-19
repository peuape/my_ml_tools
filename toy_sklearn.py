import numpy as np

# checks if the input is a numpy array. Makes sure the array is 2-dimensional
def check_type(X):
    if type(X) != np.ndarray:
        raise TypeError("The input has to be a numpy array.")
    if len(X.shape) == 1:
        return X.reshape(-1, 1)
    return X

class linear_model:
# linear regression class
    class LinearRegression:
        def __init__(self, fit_intercept=True):
            self.coef_ = None
            self.intercept_ = None
            self.w = None
            self.fit_intercept = fit_intercept

        def fit(self, X, y):
            X = check_type(X)
            y = check_type(y)
            if self.fit_intercept:
                X = np.hstack([np.ones((X.shape[0], 1)), X])

            self.w = np.dot(np.linalg.pinv(X), y) #Calculate MLE for w

            if self.fit_intercept:
                self.coef_ = self.w[1:]
                self.intercept_ = self.w[0]               
            else:
                self.coef_ = self.w
            return self


        def predict(self, X):
            X = check_type(X)
            if self.fit_intercept:  
                X = np.hstack([np.ones((X.shape[0], 1)), X])
            return np.dot(X, self.w)
    
        def score(self, X, y):
            X = check_type(X)
            y = check_type(y)
            residual_ss = np.sum((self.predict(X) - y)**2)
            total_ss = np.sum((y - np.mean(y))**2)
            return 1 - residual_ss/total_ss
            

# class for polynomial transformation
class preprocessing:
    class PolynomialFeatures:
        def __init__(self, degree, include_bias=True):
            self.degree = degree
            self.include_bias = include_bias

        def transform(self, X):
            X = check_type(X)
            if self.include_bias:
                X_poly = np.hstack([X ** i for i in range(self.degree + 1)]) 
                return X_poly
            else:
                return np.hstack([X ** i for i in range(1, self.degree + 1)])      

class model_selection:
    def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True):
        if test_size==None:
            if train_size==None:
                test_size = 0.25 
            elif 0<train_size and train_size<1:
                test_size = 1 - train_size
            else:
                raise Exception("The range of train_size is (0,1).")
        else:
            if 0<test_size and test_size<1:
                pass
            else:
                raise Exception("The range of test_size is (0,1).")
        
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
            print(combined_data)
            if shuffle == True:
                np.random.shuffle(combined_data)
            shuffled_data1 = combined_data[:,:-1*data2.shape[1]] #Separate the data again
            shuffled_data2 = combined_data[:,-1*data2.shape[1]].reshape(-1,1)  #Separate the data again
            data1_test = shuffled_data1[:test_data_len]
            data1_train = shuffled_data1[test_data_len:]
            data2_test = shuffled_data2[:test_data_len]
            data2_train = shuffled_data2[test_data_len:]
            return data1_train, data1_test, data2_train, data2_test
  
    def cross_val_score(estimator, X, y, cv=None, shuffle=False, random_state=None):
        model = estimator() #instantiate the model
        if cv == None:
            cv = 5

        if len(X)<cv:
            raise Exception("The data size has to be larger than cv.")

        combined_data = np.hstack([X, y]) #Combine data1 and data2 to shuffle them in a synchronised fashion when shuffle==True
        if shuffle==True:
            np.random.seed(random_state)
            np.random.shuffle(combined_data) 
            X = combined_data[:,:-1*y.shape[1]] 
            y = combined_data[:,-1*y.shape[1]].reshape(-1,1)
        
        scores_list = [] #The list to store the scores for each train-test pair.
        for i in range(cv):
            if i < cv-1:
                X_test = X[round(len(X)*i/cv):round(len(X)*(i+1)/cv)].reshape(-1,1)
                X_train = np.array(list(filter(lambda x: x not in X_test, X))).reshape(-1,1)
                y_test = y[round(len(y)*i/cv):round(len(y)*(i+1)/cv)].reshape(-1,1)
                y_train = np.array(list(filter(lambda y: y not in y_test, y))).reshape(-1,1)
            else:
                X_test = X[round(len(X)*(cv-1)/cv):]
                X_train = X[:round(len(X)*(cv-1)/cv)]
                y_test = y[round(len(y)*(cv-1)/cv):].reshape(-1,1)
                y_train = y[:round(len(y)*(cv-1)/cv)].reshape(-1,1)
            score = float(model.fit(X_train, y_train).score(X_train, y_train)) #fit model to data and calculate score
            scores_list.append(score)

        return scores_list
                
                
