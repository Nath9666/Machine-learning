from test import EvalModel
from sklearn.neighbors import KNeighborsRegressor
from main import df_XY_DE,df_XY_FR,df_XY_FR_features,df_XY_DE_features,dfNew_DE,dfNew_FR
from sklearn.model_selection import GridSearchCV

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
ModelKnnRegressorDE = KNeighborsRegressor(n_neighbors=6)
ModelKnnRegressorFR = KNeighborsRegressor(n_neighbors=6)

#Entrainement du model de DecisionTreeRegressor
ModelKnnRegressorDE.fit(X_train_DE,y_train_DE)
ModelKnnRegressorFR.fit(X_train_FR,y_train_FR)

#Affichage du score => coefficient de determination
print(ModelKnnRegressorDE.score(X_train_DE,y_train_DE))
print(ModelKnnRegressorFR.score(X_train_FR,y_train_FR))

#Prediction des données de test
y_pred_DE = ModelKnnRegressorDE.predict(X_test_DE)
y_pred_FR = ModelKnnRegressorFR.predict(X_test_FR)

#Evaluation du model
print('---Eval du model pour l allemagne')
EvalModel(y_pred_DE,y_test_DE)
print('---Eval du model pour la france')
EvalModel(y_pred_FR,y_test_FR)

print('\n---Changement des hyperparametre\n')


# définition des hyperparamètres à tester
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10,20,30],
    'p':[1,2,3]
}

# recherche par validation croisée pour trouver les meilleurs hyperparamètres
grid_search_DE = GridSearchCV(ModelKnnRegressorDE, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_FR = GridSearchCV(ModelKnnRegressorFR, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search_DE.fit(X_train_DE, y_train_DE)
grid_search_FR.fit(X_train_FR, y_train_FR)


print("================================")
print("Meilleurs hyperparamètres:", grid_search_DE.best_params_)
print("Meilleure performance (RMSE) :", (grid_search_FR.best_score_))
print("=================================")
print("Meilleurs hyperparamètres:", grid_search_FR.best_params_)
print("Meilleure performance (RMSE) :", (grid_search_FR.best_score_))

#Affichage du score => coefficient de determination
print(grid_search_DE.score(X_train_DE,y_train_DE))
print(grid_search_FR.score(X_train_FR,y_train_FR))

#Prediction des données de test
y_pred_DE = grid_search_DE.predict(X_test_DE)
y_pred_FR = grid_search_FR.predict(X_test_FR)

#Evaluation du model
print('\n---Eval du model pour l allemagne')
EvalModel(y_pred_DE,y_test_DE)
print('\n---Eval du model pour la france')
EvalModel(y_pred_FR,y_test_FR)


#Prediction des donnée de newX
y_pred_DE = ModelKnnRegressorDE.predict(X_New_DE)
y_pred_FR = ModelKnnRegressorFR.predict(X_New_FR)
print("y_predDE",y_pred_DE)
print("y_predFR",y_pred_FR)
