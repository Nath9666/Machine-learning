import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from main import df_XY_DE,df_XY_FR,df_XY_FR_features,df_XY_DE_features,dfNew_DE,dfNew_FR

# données d'exemple
X = df_XY_DE[df_XY_DE_features]
y = df_XY_DE["TARGET"]
x_new = dfNew_DE[df_XY_DE_features]

# création du modèle
modelLineaire = LinearRegression()

# entraînement du modèle
result = modelLineaire.fit(X, y)

# prédictions sur de nouvelles données
y_new = modelLineaire.predict(x_new)
print(y_new,y)
print('Coefficient de la pente :', modelLineaire.coef_)
print('Ordonnée à l\'origine :', modelLineaire.intercept_)
print('Prédiction pour x_new :', y_new)


#affichage des point et de la courbes
fig = plt.figure(figsize=(14,9))
#plt.scatter(x_new,y_new)
plt.plot(x_new,y_new)
plt.xlabel("New Point")
plt.ylabel("Predict point")
plt.show()