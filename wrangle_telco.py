
from env import host, user, password, get_db_url
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import os

def get_telco_data(use_cache=True):
    '''
    This function takes in no arguments, uses the imported get_db_url function to establish a connection 
    with the mysql database, and uses a SQL query to retrieve telco data creating a dataframe,
    The function caches that dataframe locally as a csv file called telco.csv, it uses an if statement to use the cached csv
    instead of a fresh SQL query on future function calls. The function returns a dataframe with the telco data.
    '''
    filename = 'telco.csv'

    if os.path.isfile(filename) and use_cache:
        print('Using cached csv...')
        return pd.read_csv(filename)
    else:
        print('Retrieving data from mySQL server...')
        df = pd.read_sql('''
        SELECT * FROM customers
        JOIN contract_types USING (contract_type_id)
        JOIN payment_types USING (payment_type_id)
        JOIN internet_service_types USING (internet_service_type_id);''' , get_db_url('telco_churn'))
        print('Caching data as csv file for future use...')
        df.to_csv(filename, index=False)
        return df



def prep_telco(telco):
    '''
    This function takes in the telco dataframe as argument, 
    converts the total_charges column to a numerical type,
    drops duplicate/unnecessary columns contract_type_id, payment_type_id, internet_service_type_id,
    encodes categorical columns from strings to binary boolean values,
    creates dummy variables for gender, contract_type, payment_type, and internet_service_type,
    renames certain columns for clarity, then returns the cleaned dataframe.
    '''

    #Dropping potential duplicates
    telco.drop_duplicates(inplace = True)
    
    #Removing leading and trailing spaces
    telco['total_charges'] = telco['total_charges'].str.strip()
    #Removing rows where tenure and total_charges = 0
    telco = telco[telco.total_charges != '']
    #Changing total_charges to appropriate data type
    telco['total_charges'] = telco.total_charges.astype(float)
   
    #Dropping duplicate columns
    columns_to_drop = ['contract_type_id', 'payment_type_id', 'internet_service_type_id']
    telco = telco.drop(columns = columns_to_drop)

    #Encoding binary categorical variables
    telco['partner'] = telco.partner.map({'Yes': 1, 'No': 0})
    telco['dependents'] = telco.dependents.map({'Yes': 1, 'No': 0})
    telco['phone_service'] = telco.phone_service.map({'Yes': 1, 'No': 0})
    telco['paperless_billing'] = telco.paperless_billing.map({'Yes': 1, 'No': 0})
    telco['churn'] = telco.churn.map({'Yes': 1, 'No': 0})
    
    #Encoding other columns to remove duplicates between phone service and internet service type
    to_replace ={'Yes': 1, 'No': 0, 'No phone service':0, 'No internet service': 0}
    telco = telco.replace(to_replace)
    
    
    #Creating dummy variables for necessary columns and concatenating to data frame, then dropping the extra columns
    #Leaving drop_first as False for initial clarity
    dummy_telco = pd.get_dummies(telco[['gender','contract_type', 'payment_type', 'internet_service_type']], 
                         dummy_na = False, 
                         drop_first = False)
    telco = pd.concat([telco, dummy_telco], axis = 1)
    telco = telco.drop(columns= ['gender', 'contract_type', 'payment_type', 'internet_service_type'])

    #renaming columns for clarity
    to_rename = {'gender_Female': 'female', 'gender_Male': 'male', 
             'contract_type_Month-to-month': 'monthly_contract',
             'contract_type_One year': 'one_yr_contract', 
             'contract_type_Two year': 'two_yr_contract', 
             'payment_type_Bank transfer (automatic)': 'auto_bank_transfer', 
             'payment_type_Credit card (automatic)': 'auto_credit_card',
             'payment_type_Electronic check': 'electronic_check',
             'payment_type_Mailed check': 'mailed_check',
             'internet_service_type_DSL': 'dsl',
             'internet_service_type_Fiber optic': 'fiber_optic',
             'internet_service_type_None': 'no_internet'}
             
    telco = telco.rename(columns=to_rename)
    
    return telco


def split_telco_data(telco):
    '''
    This function takes in the cleaned telco dataframe as argument,
    and splits the dataframe into a train, validate, and test sample,
    The train is 56% of the data, validate is 24% of the data, and test is 20% of the data,
    The function returns 3 dataframes in the order train, validate, test."
    
    '''
    train_validate, test = train_test_split(telco, test_size=.2,
                                       random_state=123,
                                       stratify=telco.churn)
    train, validate = train_test_split(train_validate, test_size=.3,
                                  random_state=123,
                                  stratify=train_validate.churn)
    return train, validate, test


def create_xy(train, validate, test):
    '''
    This functions takes in the a dataframe split into 3 train, validate, test as argumnet,
    and creates x and y variables that are subgroups for each dataframe. For the x customer_id and our target variable of churn is dropped,
    for the y only our target of churn is included.'''
    X_train = train.drop(columns=['churn', 'customer_id'])
    y_train = train.churn
    
    X_validate = validate.drop(columns=['churn', 'customer_id'])
    y_validate = validate.churn

    X_test = test.drop(columns=['churn', 'customer_id'])
    y_test = test.churn
    return X_train, y_train, X_validate, y_validate, X_test, y_test 

