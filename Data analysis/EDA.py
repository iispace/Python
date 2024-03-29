import json 
import pandas as pd

###############################################################
# Read data from text file 
# in which each line of text is formatted as json structure
###############################################################
test_file = r"experiment_log.json"

data_lst = []
with open(test_file, "r") as file:
    for line in file:
        data_lst.append(line)

jsondata_list = []
for i, line in enumerate(data_lst):
    data_dict = {}
    line_dict = json.loads(line)
    AIActAs = line_dict['AIActAs'].replace(" ", "").replace("setting","")
    mok = i // 2
    seq = f'Q{mok+1}' 
    
    data_dict.update({"TestID": seq})
    data_dict.update({"Category": AIActAs})
    data_dict.update(line_dict)
    jsondata_list.append(data_dict)


###############################################################
# convert the list of json data to DataFrame object
###############################################################
df = pd.DataFrame(jsondata_list)
df.drop(["AIActAs", "chat"], axis=1, inplace=True) # remove unneccessary columns
print(df.info())
df.head(2)

# Check if there is any row having 0(zero) value in the column, "ScoreEval".
zero_rows = df[df['ScoreEval'] == 0]
print(zero_rows)

# get sum, mean, and median value of 'ScoreEval' column 
summary_ScoreEval = df.groupby('Category').agg({'ScoreEval': ['sum', 'mean', 'median']})
display(summary_ScoreEval)



###############################################################
# Split df into two different dataframes for comparison
###############################################################
group1 = df[df['Category'] == '코드생성']['ScoreEval']    # pandas.core.series.Series
group2 = df[df['Category'] == 'Playground']['ScoreEval']  # pandas.core.series.Series



###############################################################
# Custom function to check distribution normality using Shapiro-Wilk test
###############################################################
import scipy.stats as stats
import numpy as np
from typing import Sequence, List
from termcolor import colored

def transform_data(transformer_name: str, data: Sequence[float]): # data: pandas.Series
    if transformer_name == 'log':
        return np.log(data)
    elif transformer_name == 'box_cox':
        transformed_data, lambda_value = stats.boxcox(data)
        return transformed_data
    elif transformer_name == 'square_root':
        return np.sqrt(data)

def check_normality(option: str, groups: List[str], alpha: float, transformer:None) -> List[Dict]: 
    # option: 'shapiro', 'normaltest'
    output = []
    transformer_name = "None"
    for group in groups:
        stat, p = 0.0, 0.0
        data = df[df['Category'] == group]['ScoreEval']
        if transformer is not None:
            transformer_name = transformer['name']
            data = transform_data(transformer_name, data)
        if option == 'shapiro':
            stat, p = stats.shapiro(data)
        elif option == 'normaltest':
            stat, p = stats.normaltest(data)
        output.append({"transformer":f"{transformer_name}","group": group,"data": data, "stat": stat, "p-value": p})

        print(f"Data tranformation: {transformer_name}\n")
        print(f"[Group {group}]:",end=" ")
        print(f"Shapiro-Wilk test statistics: {stat},", end=" ")
        print(f"p-value: {p} ( ", end="")
        print("{:.10f}".format(p), ")")

        if p > alpha:
            text = colored("The data appears to be normaliy distributed.", 'yellow')
            print(text)
        else:
            print("The data does NOT appear to be normaliy distributed.")
        print("="*120)
    
    return output


###############################################################
# Check distribution normality using two different methods, that are 
# shapiro-wilk test and normaltest using skewness and kurtosis.
# Apply three different transformation techniques 
# (log transformation, box-cox transformation, and squre-root transformation)
# to improve level of normal distribution when the original data distribution 
# is far from normal distribution
###############################################################
from termcolor import colored

alpha = 0.05
groups = df['Category'].unique()

transformers = [None, {'name': 'log'}, {'name': 'box_cox'}, {'name': 'square_root'}]
transformed_groups = []
test_tech = ['shapiro', 'normaltest']

for i,transformer in enumerate(transformers):
    for tech in test_tech:
        text = colored(f"[TEST TECH: {tech}]", 'green')
        print(text)
        # return: a list of dict formatted with {"transformer":str,"group": str, "data": pd.Series, "stat": float, "p-value": float}
        transformed_data = check_normality(tech, groups, alpha, transformer) 
    transformed_groups.append(transformed_data)



###############################################################
# Custom function to boxplot for multiple groups of data in one fig.
###############################################################
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.font_manager as fm 

# [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]
plt.rc('font', family='NanumGothic') # 한글깨짐 해결
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 기호 깨짐 해결
 
def boxplotting_allinone(transformed_groups: List, column_name: str, figsize=(6,8)):
    nrows = len(transformed_groups)
    fig, axes = plt.subplots(nrows=nrows, ncols=1,  figsize=figsize )

    for i, transform_group in enumerate(transformed_groups):
        transformer_name = 'Original sample data'
        if transform_group[0]['transformer'] is not None:
            transformer_name = transform_group[0]['transformer'] + " transformed data"
        
        data1 = transform_group[0]
        data2 = transform_group[1]

        axes[i].boxplot(data1['data'], positions=[1], widths=0.6) # data1.data: pandas.series
        axes[i].boxplot(data2['data'], positions=[2], widths=0.6)
        axes[i].set_xticks([1,2])
        axes[i].set_xticklabels([f"{data1['group']}", f"{data2['group']}"])
        axes[i].set_title(f"{transformer_name}") #data1.group: 코드생성 or Playground 

        axes[i].set_ylabel(f"{column_name}")

    plt.tight_layout()
    plt.show()


###############################################################
# boxplotting for multiple groups of data in one fig.
###############################################################
boxplotting_allinone(transformed_groups, 'ScoreEval', figsize=(4,7))



###############################################################
# Custom function to histplot for multiple groups of data in one fig.
###############################################################
from typing import List
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns
import matplotlib.font_manager as fm 

# [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]
plt.rc('font', family='NanumGothic') # 한글깨짐 해결
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 기호 깨짐 해결

def histogram_allinone(transformed_groups: List, column_name: str, figsize=(6,8)):
    nrows = len(transformed_groups)
    fig = plt.figure(figsize=figsize )

    for i, transform_group in enumerate(transformed_groups):
        n = (i*2) + 1 
        ax_title = fig.add_subplot(nrows, 1, i+1)
        ax1 = fig.add_subplot(nrows, 2, n)
        ax2 = fig.add_subplot(nrows, 2, n+1, sharey=ax1)

        transformer_name = 'Original sample data'
        if transform_group[0]['transformer'] is not None:
            transformer_name = transform_group[0]['transformer'] + " transformed data"
        
        data1 = transform_group[0]
        data2 = transform_group[1]

        sns.histplot(data1['data'], kde=True, ax=ax1)
        sns.histplot(data2['data'], kde=True, ax=ax2)

        ax1.set_xlabel([f"{data1['group']}"])
        ax1.set_ylabel("Count")
        ax_title.set_title(f"{transformer_name}") #data1.group: 코드생성 or Playground 
        ax_title.set_xticks([])
        ax_title.set_yticks([])

        ax2.set_xlabel([f"{data2['group']}"])
        ax2.set_ylabel("")

    plt.tight_layout()
    plt.show()

###############################################################
# histplotting for multiple groups of data in one fig.
###############################################################
histogram_allinone(transformed_groups, 'ScoreEval', figsize=(6,7))

