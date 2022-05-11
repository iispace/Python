"""
The core of this code snippet is a part of very helpful article published in kaggle, 
https://www.kaggle.com/code/entropii/timeseries-analysis-a-complete-guide-0ee2ca/edit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = np.array([1,1,0,1, 1,0,1,0, 1,1,0,0]).reshape(3,4)
print(data.shape)

df = pd.DataFrame(data=data, columns=['a','b','c','d'])

for column in df.columns:
  df[column] = df[column].replace(0, np.nan)

print(df)

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3))
sns.heatmap(df.T.isna(), cmap='Blues')
for tick in ax.yaxis.get_major_ticks():
  tick.label.set_fontsize(14)

plt.show()
                  
