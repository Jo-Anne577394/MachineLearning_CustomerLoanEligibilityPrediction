import dash
from dash import html, dcc, Input, Output
import pandas as pd
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder

import joblib

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Define layout

input_ids = [
    "Loan_ID", "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"
]
app.layout = html.Div([
    html.H1("Loan Application"),
    html.Div([
        html.Label("Loan ID"),
        dcc.Input(id='Loan_ID', type='text')
    ]),
    html.Div([
        html.Label("Gender"),
        dcc.Dropdown(
            id='Gender',
            options=[
                {'label': 'Female', 'value': 'Female'},
                {'label': 'Male', 'value': 'Male'}
            ],
            value=''  # Set default value
        )
    ]),
    html.Div([
        html.Label("Married"),
        dcc.Dropdown(
            id='Married',
            options=[
                {'label': 'Yes', 'value': 'Yes'},
                {'label': 'No', 'value': 'No'}
            ],
            value=''  # Set default value
        )
    ]),
    html.Div([
        html.Label("Dependents"),
        dcc.Dropdown(
            id='Dependents',
            options=[
                {'label': '0', 'value': '0'},
                {'label': '1', 'value': '1'},
                {'label': '2', 'value': '2'},
                {'label': '3+', 'value': '3+'}
            ],
            value=''  # Set default value
        )
    ]),
    html.Div([
        html.Label("Education"),
        dcc.Dropdown(
            id='Education',
            options=[
                {'label': 'Graduate', 'value': 'Graduate'},
                {'label': 'Not Graduate', 'value': 'Not Graduate'}
            ],
            value=''  # Set default value
        )
    ]),
    html.Div([
        html.Label("Self-Employed"),
        dcc.Dropdown(
            id='Self_Employed',
            options=[
                {'label': 'Yes', 'value': 'Yes'},
                {'label': 'No', 'value': 'No'}
            ],
            value=''  # Set default value
        )
    ]),
    html.Div([
        html.Label("Applicant Income"),
        dcc.Input(id='ApplicantIncome', type='text')
    ]),
    html.Div([
        html.Label("Coapplicant Income"),
        dcc.Input(id='CoapplicantIncome', type='text')
    ]),
    html.Div([
        html.Label("Loan Amount"),
        dcc.Input(id='LoanAmount', type='text')
    ]),
    html.Div([
        html.Label("Term"),
        dcc.Input(id='Loan_Amount_Term', type='text')
    ]),
    html.Div([
        html.Label("Credit History"),
        dcc.Input(id='Credit_History', type='text')
    ]),
    html.Div([
        html.Label("Property Area"),
        dcc.Dropdown(
            id='Property_Area',
            options=[
                {'label': 'Urban', 'value': 'Urban'},
                {'label': 'Rural', 'value': 'Rural'},
                {'label': 'Semiurban', 'value': 'Semiurban'}
            ],
            value='Male'  # Set default value
        )
    ]),
    
    
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='output-div')
])

# Callback to handle button click and write to CSV
@app.callback(
    Output('output-div', 'children'),
    [Input('submit-val', 'n_clicks')],
    [Input(id, 'value') for id in input_ids]
)
def update_output(n_clicks, *inputs):
    if n_clicks > 0:
        data = {id: [val] for id, val in zip(input_ids, inputs)}
        df = pd.DataFrame(data)
        df['Loan_ID'] = df['Loan_ID'].astype(object)
        df['Gender'] = df['Gender'].astype(object)
        df['Married'] = df['Married'].astype(object)
        df['Dependents'] = df['Dependents'].astype(object)
        df['Education'] = df['Education'].astype(object)
        df['Self_Employed'] = df['Self_Employed'].astype(object)
        df['ApplicantIncome'] = df['ApplicantIncome'].astype(int)
        df['CoapplicantIncome'] = df['CoapplicantIncome'].astype(float)
        df['LoanAmount'] = df['LoanAmount'].astype(float)
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype(float)
        df['Credit_History'] = df['Credit_History'].astype(float)
        df['Property_Area'] = df['Property_Area'].astype(object)
        # Append data to CSV file
        df.to_csv('../MLG382 Projects/Machine-Learning-382-Project-1/data/loan_applications.csv', mode='a', header=True, index=False)
        
        cleaned_data = prepare_data('../MLG382 Projects/Machine-Learning-382-Project-1/data/loan_applications.csv')
        
        return make_prediction(cleaned_data)

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
    
    df_raw_new1['Total_Income'] = df_raw_new1['ApplicantIncome'] + df_raw_new1['CoapplicantIncome']
    df_raw_new1['Loan_Term_Category'] = pd.cut(df_raw_new1['Loan_Amount_Term'], bins=[0, 180, 360, 600], labels=['Short-term', 'Medium-term', 'Long-term'])
    df_raw_new1['Income_Stability'] = df_raw_new1[['ApplicantIncome', 'CoapplicantIncome']].std(axis=1)
    df_raw_new1['Loan_to_Income_ratio'] = df_raw_new1['LoanAmount'] / ((df_raw_new1['ApplicantIncome'] + df_raw_new1['CoapplicantIncome']))
    
    df_raw_new1['Loan_Term_Category'] = df_raw_new1['Loan_Term_Category'].astype(object)
        
    ohe = OneHotEncoder(
        use_cat_names=True, 
        cols=['Gender','Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History','Property_Area', 'Loan_Term_Category']
        )

    encoded_df = ohe.fit_transform(df_raw_new1)

    return encoded_df
    
def make_prediction(data):
    prediction_data = data
    model = joblib.load('../MLG382 Projects/Machine-Learning-382-Project-1/artifacts/model2.pkl')
    
    scaler = StandardScaler()
    scaler.fit(prediction_data) 

    scaled_input = scaler.transform(prediction_data)

    predictions = model.predict(scaled_input)
    
    predictions_arr = []

    for status in predictions:
        if status > 0.5:
            predictions_arr.append('Yes')
        else:
            predictions_arr.append('No')

    print(predictions_arr)

if __name__ == '__main__':
    app.run_server(debug=True)
