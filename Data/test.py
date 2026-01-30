import numpy as np

class LogisticRegression:
    def __init__(self,lr=0.1,epochs=1000):
        self.lr=lr
        self.epochs = epochs
        self.w = None
        self.bias = None

    def sigmoid(self,z):
        pass

    def fit(self,X,y):
        N,D = X.shape
        self.w = np.zeros(D)
        self.bias = 0.0

        for epoch in range(self.epochs):
            #forward
            z = X@self.w + self.bias
            y_hat = self.sigmoid(z)

            #loss
            loss = -1/N * np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

            #backward
            dw = 1/N * X.T @ (y_hat-y)
            db = 1/N * np.sum(y_hat-y)

            #update
            self.w -= self.lr * dw
            self.bias -= self.lr * db
            if epoch % 100 == 0:
                print(f"epoch: {epoch}, loss: {loss}")

        def predict_proba(self,X):
            return self.sigmoid(X@self.w + self.bias)

        def predict(self,X,threshold=0.5):
            return (self.predict_proba(X) >= threshold).astype(int)



