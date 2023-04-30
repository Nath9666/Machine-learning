#Import des différentes librairie que l'on à besoin
import pandas as pd
import numpy as np
import numpy as np #linear algebra
import seaborn as sns #data visualization
import matplotlib.pyplot as plt #data visualization
#import sklearn.preprocessing as skp #machine learning (preprocessing)
#import sklearn.cluster as skc #machine learning (clustering)


#Recupération des données des csv
Df_dataX = pd.read_csv("Doc\Data_X.csv")
Df_dataNewX = pd.read_csv("Doc\DataNew_X.csv")
Df_dataY = pd.read_csv("Doc\Data_Y.csv")
DfMerge_DataXY = pd.merge(Df_dataX,Df_dataY,on="ID")

# Todo Suppresion de la colonne ID dans le DataFrame DfMerge_DataXY


"""
#Fusion les tableaux X et new X
cpt=0
print(dataNext.shape)
print("Df_dataY shape",Df_dataY.shape)
for ID_data2 in Df_dataY['ID']:
    find = False
    for ID_dataNext in dataNext['ID']:
        if ID_data2 == ID_dataNext:
            find = True
            break
    if not find:
        Dim = dataNext.shape[0]
        dataNext.loc[Dim] = Df_dataY.loc[cpt]
    cpt+=1
print("Data Next:",dataNext.shape)
"""

"""
#2 etape fusionner les data avec le Y grace à l'ID
merge_data1_Y = pd.merge(Df_dataX,Df_dataY,on="ID")
merge_data2_Y = pd.merge(Df_dataY,Df_dataY,on="ID")
print("Merge Df_dataX et Y:",merge_data1_Y.shape)
print("Merge Df_dataY et Y:",merge_data2_Y.shape)
"""

#Definition des différentes fonctions
def VerifNullValueByCollumns(data,nameCollumns):
    pourcent=0
    Dimension = data[nameCollumns].shape[0]
    for value in data[nameCollumns].isnull():
        if value:
            pourcent=+1
    return pourcent
def VerifNullInDataFrameByCollumns(Data):
    listPourcent = []
    for NameCollumns in Data.columns:
        listPourcent.append(VerifNullValueByCollumns(Data,NameCollumns))
    return listPourcent
def VerifyNull_RowValue(data,id):
    tuple = [0,0]
    for value in data.iloc[id].isnull():
        if value:
            tuple[0]+=1
        else:
            tuple[1]+=1
    return tuple
def VerifNull_Data(data):
    return data.isnull()
def NumberOfMisingValue(data):
    return data.isnull().sum()

#Definition d'une sous partie de mon tableau => limitation de ma machine
dfTest = DfMerge_DataXY.head(10)
#Print les info du dataX
#! print(dfTest.info)

#Print les donnees des 10 premiere lignes
#!print(dfTest.describe())

#Afficher la sommes des valeurs nulles en fonctions des collones
#!print(dfTest.isnull().sum()/dfTest.shape[0]*100)

#Afficher les valeurs dupliquers
#!print(dfTest[dfTest.duplicated(keep=False)])

#Analyse exploratoire des données

    #todo Histogramme des distributions de leur plage de valeurs

#Definition du graphiques
main = plt.figure(figsize=(12,16))
for num,collums_name in enumerate(dfTest.describe().columns):
    plt.subplot(5,2,num+1)
    sns.distplot(x=dfTest[collums_name])
    plt.xlabel(collums_name)
    plt.title('{} Distribution'.format(collums_name))
    # plt.subplots_adjust(wspace=.2, hspace=.5)
    plt.tight_layout()

test = "DE_CONSUMPTION"
#plt.subplot(5,2,4)

#sns.displot(x=dfTest[test])
plt.xlabel(test)
plt.title('{} Distribution'.format(test))
#plt.subplots_adjust(wspace=.2, hspace=1)
plt.tight_layout()
#dfTest[test].plot(kind='kde')
dfTest[test].plot(kind="hist")
plt.figure
plt.show()


#todo varaible caractéristique (collonne target) et variable cible(les celle decidé)
    #hitpogramme voir au dessus
    #diagramme en boite
    #graphique de dispersion

#todo correlation des varaibles entre elles 
#todo correlation entre le colonnes choisie et Target

#todo definir clairement ou sont les données
