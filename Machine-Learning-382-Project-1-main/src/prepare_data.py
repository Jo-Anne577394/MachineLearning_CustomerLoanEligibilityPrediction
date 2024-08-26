import pandas as pd
from category_encoders import OneHotEncoder

def prepare_data(data):
    df_raw_new = pd.read_csv(data)
    
    if 'Loan_ID' in df_raw_new.columns:
        df_raw_new.drop(columns = 'Loan_ID', inplace=True)

    df_raw_new['Credit_History'] = df_raw_new['Credit_History'].astype(object)
    missing_values = (
        df_raw_new.isnull().sum()/len(df_raw_new)*100
    ).round(0).astype(int)

    df_raw_new['LoanAmount'].fillna(df_raw_new['LoanAmount'].mean(), inplace=True)
    df_raw_new['Loan_Amount_Term'].fillna(df_raw_new['Loan_Amount_Term'].mode()[0], inplace=True)
    df_raw_new['Credit_History'].fillna(df_raw_new['Credit_History'].mode()[0], inplace=True)
    
    df_raw_new['Gender'].fillna(df_raw_new['Gender'].mode()[0], inplace=True)
    df_raw_new['Married'].fillna(df_raw_new['Married'].mode()[0], inplace=True)
    df_raw_new['Dependents'].fillna(df_raw_new['Dependents'].mode()[0], inplace=True)
    df_raw_new['Self_Employed'].fillna(df_raw_new['Self_Employed'].mode()[0], inplace=True)

    min_mask = lambda col, val: df_raw_new[col] < val
    income_mask = min_mask('ApplicantIncome',10000)
    coapplicant_income_mask = min_mask('CoapplicantIncome',5701)
    loan_ammount_mask = min_mask('LoanAmount',260)

    df_raw_new1 = df_raw_new[income_mask & coapplicant_income_mask & loan_ammount_mask]

    if 'Loan_Status' in df_raw_new1.columns:
        df_raw_new1['Loan_Status'] = df_raw_new1['Loan_Status'].apply(lambda s: 1 if s == 'Y' else 0)
        
    ohe = OneHotEncoder(
    use_cat_names=True, 
    cols=['Gender', 'Dependents', 'Married', 'Education', 'Self_Employed', 'Credit_History','Property_Area']
    )

    encoded_df = ohe.fit_transform(df_raw_new1)

    return encoded_df







