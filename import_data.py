import pandas as pd
import numpy as np

def setup():
    Df_dataX = pd.read_csv("Doc\Data_X.csv")
    Df_dataNewX = pd.read_csv("Doc\DataNew_X.csv")
    Df_dataY = pd.read_csv("Doc\Data_Y.csv")
    Df_dataMerge = pd.read_csv("Doc\Data_X.csv")
    return Df_dataX,Df_dataNewX,Df_dataY,Df_dataMerge

export 