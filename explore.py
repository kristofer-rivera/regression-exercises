
def plot_variable_pairs(df):
    sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}})

def month_to_years(df):
    df['tenure_years'] = round(df.tenure/ 12, 0)
    df['tenure_years'] = df.tenure_years.astype(int)
    return df