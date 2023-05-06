import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from main import df_XY_DE,df_XY_FR,df_XY_FR_features,df_XY_DE_features,dfNew_DE,dfNew_FR
from test import EvalModel

# données d'exemple
X = df_XY_DE[df_XY_DE_features]
y = df_XY_DE["TARGET"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# création du modèle
modelLineaire = LinearRegression()

# entraînement du modèle
modelLineaire.fit(X_train, y_train)

#score du modele
print(modelLineaire.score(X_train, y_train))

# prédictions sur de nouvelles données
y_predict = modelLineaire.predict(X_test)
print('Coefficient de la pente :', modelLineaire.coef_)
print('Ordonnée à l\'origine :', modelLineaire.intercept_)

EvalModel(y_test,y_predict)

#affichage des point et de la courbes
fig = plt.figure(figsize=(14,9))
#plt.scatter(x_new,y_new)
plt.plot(X_test,y_predict)
plt.xlabel("New Point")
plt.ylabel("Predict point")
plt.show()