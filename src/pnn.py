class PNN:
    
    pnn_bypass = True
    
    def __init__(self, sigma):
        self.sigma = sigma

    def __repr__(self):
        return 'Sigma(%r)' % self.sigma
    
    def fit(self,X_train,X_test,y_train):
        # n_classes = len(my_data.groupby('<class_name_space>''))
        ## TODO: improve the number of classes value
        sigma = self.sigma
        n_classes = max(y_train)
        prob = np.zeros((X_test.shape[0],max(y_train))) 
    #     print(prob.shape)
        # X_test = np.array([[0.5,0.5],[0.8,0.2],[0.4,0.7]])
        # prob = np.zeros((X_test.shape[0],max(y)[0]))

        ## loop through all the X_test data (unclassified points)
        for point in range(0,X_test.shape[0]):
            x_test = X_test[point,:] 
        #     print(x_test)
            ## loop through i-th class
            for i in range(1,n_classes+1):
                X_i = X_train[np.where(y_train == i)[0],:]
                summ = 0.
                ## loop through j-th element of the i-th class
                for j in range(0,X_i.shape[0]):
                    ## norm L2 (sum of squares)
                    norm = 0.
                    ## loop through k-th feature
                    for k in range(0,X_i.shape[1]):
                        ## TODO: covariance form X*M*X'
                        norm = norm + (X_i[j,k]-x_test[k])*(X_i[j,k]-x_test[k])  
                    ## Summation of Gaussians
                    summ = summ + np.exp((-1/2)*(1/(sigma**2))*norm)
                ## Average of Parzen Window (normalization term)
                summ = summ/X_i.shape[0]
                ## Decision boundary
                prob[point, (i-1)] = summ
                self.prob = prob
                self.X_train = X_train
                self.y_train = y_train
                self.X_test = X_test
                
    def score(self, X_test,y_test):
        y_prob = np.asarray([max(self.prob[i,:]) for i in range(0,self.prob.shape[0])])
        self.y_bar = np.asarray([np.where(self.prob[ix,:] == max(self.prob[ix,:]))[0][0]+1 for ix in range(0,self.prob.shape[0])])
        acc_score = accuracy_score(y_test, self.y_bar)
        
        return acc_score

    def predict(self):
        ## TODO: 
        # argmax of each 
#         prob = self.prob
        y_prob = np.asarray([max(self.prob[i,:]) for i in range(0,self.prob.shape[0])])
        self.y_bar = np.asarray([np.where(prob[ix,:] == max(prob[ix,:]))[0][0]+1 for ix in range(0,prob.shape[0])])
#         self.y_bar = y_bar
#         return self.y_bar
        
    
    def predict_proba(self,X_range):
        self.fit(self.X_train,X_range,self.y_train)
        
        return self.prob