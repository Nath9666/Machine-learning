import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization
from sklearn.linear_model import LinearRegression
from VerifyData import NumberOutiliners


#Recupération des données des csv
Df_dataX = pd.read_csv("Doc\Data_X.csv")
Df_dataNewX = pd.read_csv("Doc\DataNew_X.csv")
Df_dataY = pd.read_csv("Doc\Data_Y.csv")

#fussionner les data X et Y
DfMerge_DataXY = pd.merge(Df_dataX,Df_dataY,on="ID")

#definition d'un data frame test pour le bien de ma machine
DF_Test = DfMerge_DataXY.head(100)

#remplacé les valeur nulle par la moyenne de la colonnes
#DF_Test.fillna(DF_Test.mean(), inplace=True)
    #todo verifiation si il n'y a plus de valeur nulles


# Créer un groupe pour chaque pays
grouped = DF_Test.groupby('COUNTRY')

# Récupérer le groupe FR dans un nouveau DataFrame
df_FR = grouped.get_group('FR')

# Récupérer le groupe DE dans un nouveau DataFrame
df_DE = grouped.get_group('DE')

#Recuperation des collones appartenant à FR et DE
ColloneGroup = ['COAL_RET', 'GAS_RET', 'ID', 'CARBON_RET', 'DAY_ID', 'COUNTRY', 'TARGET']
DE_cols = [col for col in df_DE.columns if col.startswith('DE') or col in ColloneGroup]
FR_cols = [col for col in df_FR.columns if col.startswith('FR') or col in ColloneGroup]

#Recuperation des donnees
df_XY_DE = df_DE.loc[:,DE_cols]
df_XY_FR = df_FR.loc[:,FR_cols]

df_XY_DE = df_XY_DE.sort_values(by='DAY_ID')
df_XY_FR = df_XY_FR.sort_values(by='DAY_ID')

#Supression de la collones id_day id et country
col = ['ID', 'COUNTRY','DAY_ID']
df_XY_DE = df_XY_DE.drop(col,axis=1)
df_XY_FR = df_XY_FR.drop(col,axis=1)

#todo verifiez qu'il n'y a pas de doublons dans les 2 tables
"""
print(df_XY_DE[df_XY_DE.duplicated(keep=False)])
print(df_XY_FR[df_XY_FR.duplicated(keep=False)])
"""

#verifiez les outliners
for col in df_XY_DE.columns:
    print(col)
    print(NumberOutiliners(df_XY_DE,col))
