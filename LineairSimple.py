from test import EvalModel
from sklearn.linear_model import LinearRegression
from main import df_XY_DE,df_XY_FR,df_XY_FR_features,df_XY_DE_features,dfNew_DE,dfNew_FR

#Initialisation des données pour l'allemagne
X_DE = df_XY_DE[df_XY_DE_features]
y_DE = df_XY_DE["TARGET"]
X_New_DE = dfNew_DE[df_XY_DE_features]

X_FR = df_XY_FR[df_XY_FR_features]
y_FR = df_XY_FR["TARGET"]
X_New_FR = dfNew_FR[df_XY_FR_features]

#Separation en donnée de test et d'entrainement
from sklearn.model_selection import train_test_split
X_train_DE, X_test_DE, y_train_DE, y_test_DE = train_test_split(X_DE, y_DE, test_size=0.2, random_state=42)
X_train_FR, X_test_FR, y_train_FR, y_test_FR = train_test_split(X_FR, y_FR, test_size=0.2, random_state=42)

#Creation du model linéaire
    #Il n'y a pas d'hyperparametre pour le model de regression simple
ModelLineaireDE = LinearRegression()
ModelLineaireFR = LinearRegression()

#Entrainement du model linéaire
ModelLineaireDE.fit(X_train_DE,y_train_DE)
ModelLineaireFR.fit(X_train_FR,y_train_FR)

#Affichage du score => coefficient de determination
print(ModelLineaireDE.score(X_train_DE,y_train_DE))
print(ModelLineaireFR.score(X_train_FR,y_train_FR))

#Prediction des données de test
y_pred_DE = ModelLineaireDE.predict(X_test_DE)
y_pred_FR = ModelLineaireFR.predict(X_test_FR)

#Evaluation du model
print('---Eval du model pour l allemagne')
EvalModel(y_pred_DE,y_test_DE)
print('---Eval du model pour la france')
EvalModel(y_pred_FR,y_test_FR)

#Réentrainement de machine sur la totalité des données
ModelLineaireDE.fit(X_DE,y_DE)
ModelLineaireFR.fit(X_FR,y_FR)

#Prediction des donnée de newX
y_pred_DE = ModelLineaireDE.predict(X_New_DE)
y_pred_FR = ModelLineaireFR.predict(X_New_FR)

#! on peu en conclure que le model n'est pas adpaté pour se gendre de données