import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization


#Recupération des données des csv
Df_dataX = pd.read_csv("Doc\Data_X.csv")
Df_dataNewX = pd.read_csv("Doc\DataNew_X.csv")
Df_dataY = pd.read_csv("Doc\Data_Y.csv")

#fussionner les data X et Y et limitation pour 200 data
DfMerge_DataXY = pd.merge(Df_dataX,Df_dataY,on="ID").head(200)
DfMerge_DatanewX = Df_dataNewX.head(200)


# Créer un groupe pour chaque pays
grouped = DfMerge_DataXY.groupby('COUNTRY')
groupedNew = DfMerge_DatanewX.groupby('COUNTRY')

# Récupérer le groupe FR dans un nouveau DataFrame
df_FR = grouped.get_group('FR')
dfNew_FR = groupedNew.get_group('FR')

# Récupérer le groupe DE dans un nouveau DataFrame
df_DE = grouped.get_group('DE')
dfNew_DE = groupedNew.get_group('DE')

#Recuperation des collones appartenant à FR et DE
ColloneGroup = ['COAL_RET', 'GAS_RET', 'ID', 'CARBON_RET', 'DAY_ID', 'COUNTRY']
DE_cols = [col for col in df_DE.columns if col.startswith('DE') or col in ColloneGroup or col =='TARGET']
FR_cols = [col for col in df_FR.columns if col.startswith('FR') or col in ColloneGroup or col =='TARGET']

DE_colsNew = [col for col in df_DE.columns if col.startswith('DE') or col in ColloneGroup]
FR_colsNew = [col for col in df_FR.columns if col.startswith('FR') or col in ColloneGroup]

#Recuperation des donnees
df_XY_DE = df_DE.loc[:,DE_cols]
df_XY_FR = df_FR.loc[:,FR_cols]

dfNew_FR = dfNew_FR.loc[:,FR_colsNew]
dfNew_DE = dfNew_DE.loc[:,DE_colsNew]


#Supression de la collones id_day id et country
col = ['ID', 'COUNTRY','DAY_ID']
df_XY_DE = df_XY_DE.drop(col,axis=1)
df_XY_FR = df_XY_FR.drop(col,axis=1)

dfNew_FR = dfNew_FR.drop(col,axis=1)
dfNew_DE = dfNew_DE.drop(col,axis=1)

#verifiez qu'il n'y a pas de doublons dans les 2 tables
"""print(df_XY_DE[df_XY_DE.duplicated(keep=False)])
print(df_XY_FR[df_XY_FR.duplicated(keep=False)])
print(dfNew_FR[dfNew_FR.duplicated(keep=False)])
print(dfNew_DE[dfNew_DE.duplicated(keep=False)])"""

#verification des valeurs nulles
"""print(df_XY_DE.isnull().sum()/df_XY_DE.shape[0]*100)
print(df_XY_FR.isnull().sum()/df_XY_FR.shape[0]*100)
print(dfNew_FR.isnull().sum()/dfNew_FR.shape[0]*100)
print(dfNew_DE.isnull().sum()/dfNew_DE.shape[0]*100)"""

#Affichage et sauvegarde des matrice de correlation

#Matrice de correlation de l'allemagne
correlation_metrics=df_XY_DE.corr()
fig = plt.figure(figsize=(14,9))
sns.heatmap(correlation_metrics,square=True, annot=True, vmax=1, vmin=-1, cmap='RdBu')
plt.title('Correlation Between Variables in DE', size=14)
plt.savefig('DE/DE_correlation.png')
#plt.show()
#plt.close(fig)


#Matrice de correlation de la France
correlation_metrics=df_XY_FR.corr()
fig = plt.figure(figsize=(14,9))
sns.heatmap(correlation_metrics,square=True, annot=True, vmax=1, vmin=-1, cmap='RdBu')
plt.title('Correlation Between Variables in DE', size=14)
plt.savefig('FR/FR_correlation.png')
#plt.show()
#plt.close(fig)

def SaveDispersionGraph(data,name,source):
    fig = plt.figure(figsize=(8,6))
    sns.scatterplot(x=data[name], y=data.TARGET)
    plt.title(data[name].name + "vs" + data.TARGET.name)
    plt.savefig(source+'/Graph_Dispersion/%s_by_%s.png' % (data[name].name,data.TARGET.name))
    plt.close(fig)
def SaveKDE(df,namecolumns,source):
    fig = plt.figure(figsize=(12,16))
    df[namecolumns].plot(kind="kde")
    plt.title('{} BoxPlot'.format(namecolumns))
    plt.savefig(source+'/Graph_KDE/%s.png' % (df[namecolumns].name))
    plt.close(fig)
def SaveHistoByCollumn(df,namecolumns,source):
    fig = plt.figure(figsize=(12,16))
    df[namecolumns].plot(kind="hist")
    plt.title('{} BoxPlot'.format(namecolumns))
    plt.savefig(source+'/Graph_Histogram/%s.png' % (df[namecolumns].name))
    plt.close(fig)
def SaveBoxPlot(df,column,source):
    fig = plt.figure(figsize=(12,16))
    df.boxplot(column) 
    plt.title('{} BoxPlot'.format(column))
    plt.savefig(source+'/Graph_BoxPlot/%s.png' % (df[column].name))
    plt.close(fig)

"""for column in df_XY_DE.columns:
    SaveDispersionGraph(df_XY_DE,column,"DE")
    SaveKDE(df_XY_DE,column,"DE")
    SaveBoxPlot(df_XY_DE,column,"DE")
    SaveHistoByCollumn(df_XY_DE,column,"DE")

for column in df_XY_FR.columns:
    SaveDispersionGraph(df_XY_FR,column,"FR")
    SaveBoxPlot(df_XY_FR,column,"FR")
    SaveKDE(df_XY_FR,column,"FR")
    SaveHistoByCollumn(df_XY_FR,column,"FR")
"""

#remplacement des valeurs nulls par la moyenne de chaque colonees
df_XY_DE.fillna(df_XY_DE.mean(), inplace=True)
df_XY_FR.fillna(df_XY_FR.mean(), inplace=True)
dfNew_DE.fillna(dfNew_DE.mean(), inplace=True)
dfNew_FR.fillna(dfNew_FR.mean(), inplace=True)

df_XY_FR_features = ["FR_CONSUMPTION","FR_DE_EXCHANGE","FR_NET_IMPORT","FR_WIND","FR_RESIDUAL_LOAD"]
df_XY_DE_features = ["DE_FR_EXCHANGE", "DE_NET_EXPORT", "DE_COAL","DE_WIND","DE_RESIDUAL_LOAD","DE_WINDPOW"]
