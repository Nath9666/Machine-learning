import pandas as pd
import numpy as np

data1 = pd.read_csv("Doc\Data_X.csv")
data2 = pd.read_csv("Doc\DataNew_X.csv")

print(data1.describe())

for id in data1['ID'].unique():
    print(id)