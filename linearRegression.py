from sklearn.linear_model import LinearRegression
import numpy as np

X_train = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
y_train = np.array([2, 4, 5, 4, 5])
x_new = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))


lin = LinearRegression()
lin.fit(X_train, y_train)
print(lin.coef_)

"""
--- knn
--- liear regression 

--- interpretation et rapport
"""