import numpy as np
class LinearRegression:
    def __init__(self,lr=0.0001,epochs=100):
        self.lr=lr
        self.epochs=epochs
        self.w=None
        self.b=None
    def cost_function(self,y,y_prediction,m):
        error= y - y_prediction
        cost= (1/m) * np.sum(error) ** 2
        return cost
    def fit(self,X,y):
        m,n=X.shape
        self.w=np.zeros((n,1))
        self.b=0

        for i in range(self.epochs):
            y_prediction= np.dot(X,self.w) + self.b
            cost=self.cost_function(y,y_prediction,m)
            print(f'Epoch : {i+1} Cost : {cost:.6f}')
            dw= (2 / m) * (np.dot(X.T,(y_prediction-y)))
            db= (2 / m) * (np.sum(y_prediction-y))
            self.w -= self.lr * dw
            self.b -= self.lr * db
    def predict(self,X):
        return np.dot(X,self.w)+self.b
         