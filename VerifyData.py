from import_data import *
import pandas as pd
import numpy as np

Df_dataX,Df_dataNewX,Df_dataY,Df_dataMerge = setup()

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

#2 etape fusionner les data avec le Y grace à l'ID
merge_data1_Y = pd.merge(Df_dataX,Df_dataY,on="ID")
merge_data2_Y = pd.merge(Df_dataY,Df_dataY,on="ID")
print("Merge Df_dataX et Y:",merge_data1_Y.shape)
print("Merge Df_dataY et Y:",merge_data2_Y.shape)

#Verification si il y a des valeurs manquantes dans les données
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


print(VerifNullInDataFrameByCollumns(Df_dataX))
print(VerifyNull_RowValue(Df_dataX,0))
print(VerifNull_Data(Df_dataX))



#3 selectioonner les sous ensemble de données

#4 creer les nouvel collone graca au ancienne collone

#5 triée les lignes en fonction des nouvelles collones