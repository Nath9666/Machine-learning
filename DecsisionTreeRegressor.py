from main import df_XY_DE,df_XY_FR,df_XY_FR_features,df_XY_DE_features,dfNew_DE,dfNew_FR
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

# Création du dataframe
df = pd.DataFrame({'x': [1, 2, 3, 4, 5],
                   'y': [2, 4, 5, 4, 6]})

# Séparation des données en ensemble d'entraînement et ensemble de test
x_train, x_test, y_train, y_test = train_test_split(df_XY_DE[df_XY_DE_features], df_XY_DE['TARGET'], test_size=0.2, random_state=0)

# Création de l'objet de modèle d'arbre de décision
model = DecisionTreeRegressor(random_state=3)

# Entraînement du modèle sur les données d'entraînement
model.fit(x_train, y_train)

# Prédiction des valeurs de y pour les données de test
y_pred = model.predict(x_test)
print(y_pred)

# Affichage des résultats
print("Correlation de Spearman : ", spearmanr(y_test, y_pred)[0])
print("Coefficient de determination R2 : ", r2_score(y_test, y_pred))
print("Erreur quadratique moyenne (RMSE) : ", mean_squared_error(y_test, y_pred, squared=False))