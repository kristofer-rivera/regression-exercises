
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_variable_pairs(df):
    sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}})

def month_to_years(df):
    df['tenure_years'] = round(df.tenure/ 12, 0)
    df['tenure_years'] = df.tenure_years.astype(int)
    return df

def plot_categorical_and_continuous_vars(df, cat_var, cont_var):
    sns.barplot(data=df, y=cont_var, x=cat_var)
    plt.show()
    sns.boxplot(data=df, y=cont_var, x=cat_var)
    plt.show()
    sns.stripplot(data=df, y=cont_var, x=cat_var)
    plt.show()