import numpy as np
from sklearn.linear_model import LinearRegression

# données d'exemple
X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
y = np.array([2, 4, 5, 4, 5])

# création du modèle
model = LinearRegression()

# entraînement du modèle
model.fit(X, y)

# prédictions sur de nouvelles données
x_new = np.array([5]).reshape((-1, 1))
y_new = model.predict(x_new)

