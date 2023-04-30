from sklearn.tree import DecisionTreeRegressor
from import_data import *

Df_dataX,Df_dataNewX,Df_dataY,Df_dataMerge = setup()

#Selection de la variable données
y = Df_dataY

#Selection des collonnes
features_name = []

#Selectionner les données dans la base
X = Df_dataX[features_name]

#

melbourne_model = DecisionTreeRegressor()