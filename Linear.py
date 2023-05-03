import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# données d'exemple
X = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
y = np.array([2, 4, 5, 4, 5])
x_new = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))

# création du modèle
modelLineaire = LinearRegression()

# entraînement du modèle
modelLineaire.fit(X, y)

# prédictions sur de nouvelles données
y_new = modelLineaire.predict(x_new)
print('Coefficient de la pente :', modelLineaire.coef_)
print('Ordonnée à l\'origine :', modelLineaire.intercept_)
print('Prédiction pour x_new :', y_new)

#affichage des point et de la courbes
fig = plt.figure(figsize=(14,9))
plt.scatter(x_new,y_new)
plt.plot(x_new,y_new)
plt.xlabel("New Point")
plt.ylabel("First Point")
plt.show()