from env import host, user, password, get_db_url
import pandas as pd 
import os




def get_zillow_raw(use_cache=True):
    '''
    This function takes in no arguments, uses the imported get_db_url function to establish a connection 
    with the mysql database, and uses a SQL query to retrieve telco data creating a dataframe,
    The function caches that dataframe locally as a csv file called zillow_raw.csv, it uses an if statement to use the cached csv
    instead of a fresh SQL query on future function calls. The function returns a dataframe with the telco data.
    '''
    filename = 'zillow_raw.csv'

    if os.path.isfile(filename) and use_cache:
        print('Using cached csv...')
        return pd.read_csv(filename)
    else:
        print('Retrieving data from mySQL server...')
        df = pd.read_sql('''
        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017
        JOIN propertylandusetype
        USING (propertylandusetypeid)
        WHERE propertylandusetypeid = 261;''' , get_db_url('zillow'))
        print('Caching data as csv file for future use...')
        df.to_csv(filename, index=False)
        return df

def wrangle_zillow():
    '''
    This function takes in no arguments, uses the get_zillow_raw function to establish a connection with the mysql database and acquire a raw version 
    of the zillow dataframe. The function then performes the necesary operations to rename columns and drop all null values returning a cleaned zillow dataframe with zero nulls.
    '''
    #get zillow data
    df = get_zillow_raw()
    #drop all nulls
    df = df.dropna()
    #convert floats to int
    df = df.astype('int')
    #renaming columns for clarity
    df = df.rename(columns = {'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 
                                  'calculatedfinishedsquarefeet': 'sqft', 'taxvaluedollarcnt': 'tax_value', 
                                  'yearbuilt': 'year_built', 'taxamount': 'tax_amount'})
    return df
