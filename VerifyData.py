#Import des différentes librairie que l'on à besoin
import pandas as pd
import numpy as np
import seaborn as sns #data visualization
import matplotlib.pyplot as plt #data visualization
from sklearn.preprocessing import MinMaxScaler
#import sklearn.preprocessing as skp #machine learning (preprocessing)
#import sklearn.cluster as skc #machine learning (clustering)


#Recupération des données des csv
Df_dataX = pd.read_csv("Doc\Data_X.csv")
Df_dataNewX = pd.read_csv("Doc\DataNew_X.csv")
Df_dataY = pd.read_csv("Doc\Data_Y.csv")
DfMerge_DataXY = pd.merge(Df_dataX,Df_dataY,on="ID")


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
dfTest = DfMerge_DataXY.head(300)

#On enlève les collones ID et ID day et contry juste pour voir les valeurs
dfTest = dfTest.drop('ID',axis=1)
dfTest = dfTest.drop('DAY_ID',axis=1)

#on enlève la collones contry car elle ne possède pas de variable quantitative
dfTest = dfTest.drop('COUNTRY',axis=1)

#Print les info du dataX
#! print(dfTest.info)

#Print les donnees des 10 premiere lignes
#! print(dfTest.describe())

#Afficher la sommes des valeurs nulles en fonctions des collones
#! print(dfTest.isnull().sum()/dfTest.shape[0]*100)

#Afficher les valeurs dupliquers
#!print(dfTest[dfTest.duplicated(keep=False)])


#*Analyse exploratoire des données

    #Histogramme des distributions des colonnes

#Fonction pour afficher le graphique de distribution en fonction d'un data frame de 8 colonne max
def AfficheHistogram(data):
    plt.figure(figsize=(12,16))
    for num,collums_name in enumerate(data.describe().columns):
        plt.subplot(4,2,num+1)
        #sns.histplot(x=df1[collums_name])
        data[collums_name].plot(kind="hist",y="distribution")
        plt.title('{} Distribution'.format(collums_name))
        #plt.subplots_adjust(wspace=.2, hspace=.5)
        plt.tight_layout()

def SaveHistoByCollumn(df,namecolumns):
    fig = plt.figure(figsize=(12,16))
    df[namecolumns].plot(kind="hist")
    plt.title('{} BoxPlot'.format(namecolumns))
    plt.savefig('Graph/Graph_Histogram/%s.png' % (df[namecolumns].name))
    plt.close(fig)

"""
#*Toute les distribution des différentes colonnes
AfficheHistogram(df1)
AfficheHistogram(df2)
AfficheHistogram(df3)
AfficheHistogram(df4)
plt.show()
"""

    #Affiche de leur plage de valeur grace au box plot

#Fonction affiche la plage de valeur en fonction d'un data frame de 8 coloonnes max
def SaveBoxPlot(df,namecolumns):
    fig = plt.figure(figsize=(12,16))
    df[namecolumns].plot(kind="kde")
    plt.title('{} BoxPlot'.format(namecolumns))
    plt.savefig('Graph/Graph_BoxPlot/%s.png' % (df[namecolumns].name))
    plt.close(fig)

def NumberOutiliners(df,colons):
    # Calculate the quartiles and IQR for column A
    Q1 = df[colons].quantile(0.25)
    Q3 = df[colons].quantile(0.75)
    IQR = Q3 - Q1

    # Identify the outliers in column A
    outliers = df[(df[colons] < Q1 - 1.5*IQR) | (df[colons] > Q3 + 1.5*IQR)]

    # Print the number of outliers
    print("Number of outliers:", len(outliers))
#todo varaible caractéristique (collonne target) et variable cible(les celle decidé)
    #graphique de dispersion
        #Recuperer la liste de chaque colonnes sans la variable target


#*Creer les differents graphiques de dispersions des données des collones avec la target
def SaveDispersionGraph(data,name):
    fig = plt.figure(figsize=(8,6))
    sns.scatterplot(x=data[name], y=data.TARGET)
    plt.title('DE_CONSUMPTION vs TARGET')
    plt.savefig('Graph/Graph_Dispersion/%s_by_%s.png' % (data[name].name,data.TARGET.name))
    plt.close(fig)
    #plt.show()

"""

for collumns_name in dfTest.columns:
    Dispersion(dfTest,collumns_name)
    SaveBoxPlot(dfTest,collumns_name)
    SaveHistoByCollumn(dfTest,collumns_name)
"""

"""
SaveBoxPlot(DfMerge_DataXY,'DE_CONSUMPTION')
print(DfMerge_DataXY.DE_CONSUMPTION.describe())
SaveHistoByCollumn(DfMerge_DataXY,'DE_CONSUMPTION')
"""

"""
for col in dfTest.columns:
    SaveBoxPlot(dfTest,col)
    SaveHistoByCollumn(dfTest,col)
    SaveDispersionGraph(dfTest,col)
"""

NumberOutiliners(dfTest,"DE_CONSUMPTION")


#todo correlation des varaibles entre elles 
"""
correlation_metrics=dfTest.corr()
print(correlation_metrics)
fig = plt.figure(figsize=(14,9))
sns.heatmap(correlation_metrics,square=True, annot=True, vmax=1, vmin=-1, cmap='RdBu')
plt.title('Correlation Between Variables', size=14)
plt.show()
"""

#todo correlation entre le colonnes choisie et Target


#todo definir clairement ou sont les données
