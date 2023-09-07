import json 
import pandas as pd

test_file = r"experiment_log.json"

# read data from textfile in which each line of text is formatted as json structure. i.e., key, value pair enclosed with curly bracket.
data_lst = []
with open(test_file, "r") as file:
    for line in file:
        data_lst.append(line)

jsondata_list = []
for i, line in enumerate(data_lst):
    new_dict = {}
    line_dict = json.loads(line)
    AIActAs = line_dict['AIActAs'].replace(" ", "").replace("setting","")
    mok = i // 2
    seq = f'Q{mok+1}' 
    
    new_dict.update({"TestID": seq})
    new_dict.update({"Category": AIActAs})
    new_dict.update(line_dict)
    jsondata_list.append(new_dict)

###############################################################
# convert the data to DataFrame object
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
# Split df into two different dataframe for comparison
###############################################################
group1 = df[df['Category'] == '코드생성']['ScoreEval']    # pandas.core.series.Series
group2 = df[df['Category'] == 'Playground']['ScoreEval']  # pandas.core.series.Series

###############################################################
# Custom function to check distribution normality using Shapiro-Wilk test
###############################################################
import scipy.stats as stats
import numpy as np
from typing import Sequence, List

def transform_data(transformer_name: str, data: Sequence[float]): # data: pandas.Series
    if transformer_name == 'log':
        return np.log(data)
    elif transformer_name == 'box_cox':
        transformed_data, lambda_value = stats.boxcox(data)
        return transformed_data

def check_normality(groups: List[str], alpha: float, transformer:None) -> List[Dict]: # transformer: None or dict object having 'name' key and its value in string format. ex: {'name': 'log'}
    output = []
    transformer_name = "None"
    for group in groups:
        data = df[df['Category'] == group]['ScoreEval']
        if transformer is not None:
            transformer_name = transformer['name']
            data = transform_data(transformer_name, data)
        stat, p = stats.shapiro(data)
        output.append({"transformer":f"{transformer_name}","group": group,"data": data, "stat": stat, "p-value": p})

        print(f"Data tranformation: {transformer_name}\n")
        print(f"[Group {group}]:",end=" ")
        print(f"Shapiro-Wilk test statistics: {stat},", end=" ")
        print(f"p-value: {p} ( ", end="")
        print("{:.10f}".format(p), ")")

        if p > alpha:
            print("The data appears to be normaliy distributed.")
        else:
            print("The data does NOT appear to be normaliy distributed.")
        print("="*120)
    
    return output

###############################################################
# Check distribution normality by calling "check_normality()" function
###############################################################
groups = df['Category'].unique()
alpha = 0.05

Check_org_data = check_normality(groups, alpha, None) # return: a list of dict formatted with {"transformer":str,"group": str, "data": pd.Series, "stat": float, "p-value": float}

###############################################################
# Apply two transformation techniques (log transformation, box-cox transformation)
# to improve level of normal distribution when the original data distribution 
# is far from normal distribution
###############################################################
transformer = {'name': 'log'}
check_log_data = check_normality(groups, alpha, transformer)

transformer = {'name': 'box_cox'}
check_boxcox_data = check_normality(groups, alpha, transformer)


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
org_dict, log_dict, boxcox_dict = [], [], []
transformed_groups = [org_dict, log_dict, boxcox_dict]
transformed_data = [check_source_data, check_log_data, check_boxcox_data]

for i, data in enumerate(transformed_data):
    for j in range(2):
        transformed_groups[i].append({"transformer":transformed_data[i][j]['transformer'], "group": transformed_data[i][j]['group'], "data": transformed_data[i][j]['data']})

boxplotting_allinone(transformed_groups, 'ScoreEval', figsize=(4,7))


