import pandas as pd
import numpy as np
class LassoRegression:
    def _init_(self,learning_rate=0.01,n_iters=1000,lambda_1=0):
        self.lr=learning_rate
        self.iterations=n_iters
        self.lambda_1=lambda_1
        self.thetha=None
        self.bias=None
        self.loss=[]
    def _linear_model(self,X):
        '''
        Compute the linear model for the given feature matrix.
        Args:
            X(numpy.ndarray):the feature matrix to predict.
        Returns:
            numpy.ndarray:the computed linear model.
        '''
        return np.dot(X,self.thetha)+self.bias
    def _initialize_parameters(self,n_features):
        self.thetha=np.random.random_sample(n_features)
        self.bias=np.random.random_sample()
    def _compute_thetha_derivative(self,X,y,linear_model):
        n_sample, n_feature=X.shape
        d_thetha=np.zeros(n_feature)
        for j in range(n_feature):
            if self.thetha[j]>0:
                d_thetja[j]=-(2/n_sample)*(np.dot(X[:,j]),(y-linear_model())+self.lambda_1)
            else:
                d_thetja[j]=-(2/n_sample)*(np.dot(X[:,j]),(y-linear_model())-self.lambda_1)
        return d_thetha
    def _compute_bias_derivative(self,X,y,linear_model):
        d_bias=-(2/X.shape[0] + np.sum(y-lineaar_model))
        return d_bias
    def _validate_inputs(self,X,y):
        assert X.shape[0]==y.shape[0],'The Number Of Samples in the feature matrix and the target value should be equal'
    def _calculate_cost(self,y,z):
        '''
        Calculate the Lasso Cost(error) for the given target,prediction and lambda.
        Args:
            y(numpy.ndarray):The True Target Values.
            z(numpy.ndarray):The Predicted Target Values.
        Returns:
            float: The calculated Lasso cosr(root mean squared error + L1 penalty)
        '''
        n_samples=y.shape[0]
        lasso_loss=(1/n_samples)+np.sum(np.square(y-z))+(self.lambda_1*np.sum(np.abs(self.thetha)))
        return lasso_loss
    def fit(self,X,y):
        self._validate_inputs(X,y)
        self._initialize_parameters(X.shape[1])
        for _ in range(self.iterations):
            linear_model=self._linear_model(X)
            d_thetha=self._compute_thetha_derivative(X,y,linear_model)
            d_bias=self._compute_bias_derivative(X,y,linear_model)
            self.thetha-=self.lr*d_thetha
            self.bias-=self.lr*d_bias
            self.loss.append(self._calculate_cost(y,linear_model))
    def predict(self,X):
        '''
        Predict the target value for the given feature matrix.
        Args:
            X(numpy.ndarray):the feature matrix to predict.
        Returns:
            numpy.ndarray:the predicted target values.
        '''
        return self._linear_model(X)
    def cost(self):
        return self.loss