#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np


# %%


#choose dataset and model from list
datasets = {0: 'census_income', 1: 'bank_marketing', 2: 'statlog_shuttle', 3: 'diabetes', 4:'gsvs', 5: 'cover_type'}
models= {0: 'lgbm', 1: 'rf', 2: 'svm'}

model = models[0]
dataset = datasets[0]


# %%


df = pd.read_csv(f'results/{model}/{model}_{dataset}_combined_importances.csv')
df['AUC'] = df['AUC'].apply(lambda x: ast.literal_eval(x))

number = 0
y_ticks = np.round(np.arange(0.1, 1.1, 0.1), 2).tolist()
xmin, xmax = round(min(min(df['AUC'])),2)-0.05, round(max(max(df['AUC'])),2)+0.1
for _ in range(int(df.shape[0]/10)):
    f_no = df['Feature_ID'].iloc[0+number]
    plt.figure()
    df['AUC'].iloc[0+number:10+number].apply(lambda x: pd.Series(x)).T.boxplot(figsize=(7, 5), vert=False,
                                                            boxprops=dict(color='blue'),
                                                            medianprops=dict(color='green'),
                                                            flierprops=dict(marker='x', markersize=6, markeredgecolor='red'),
                                                            whiskerprops=dict(linestyle='--', dashes = (10, 5)))
    
    plt.rc('font', family='Times New Roman')
    plt.xlabel('Decrease in AUC',fontsize=14)
    plt.ylabel('Sample Fractions',fontsize=14)
    plt.yticks(ticks=np.arange(1, 11), labels=y_ticks)

    plt.xlim([xmin, xmax])
    plt.axvline(x=0, color='black', linestyle='-.')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig(f'results/{model}/{model}_{dataset}_feature_{f_no}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
    number += 10


# %%




