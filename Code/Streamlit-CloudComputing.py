import streamlit as st
import pandas as pd
import sagemaker
from sklearn.preprocessing import LabelEncoder
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker import get_execution_role
from sagemaker.serializers import CSVSerializer
import boto3

# Set the region
region_name = 'us-east-1'
boto3.setup_default_session(region_name=region_name)
sagemaker_session = sagemaker.Session()
role = get_execution_role()

st.set_page_config(page_title="Team 3 Cloud Computing Project", page_icon=":tada:", layout="wide")

st.subheader("This is project to predict selling price of used cars")

le = LabelEncoder()

#Inputs from user
odometer_value = st.number_input("Odometer Value", min_value=0)
engine_type = st.selectbox("Engine Type", ['Gas', 'Diesel', 'Electric'])
engine_fuel_type = st.selectbox("Engine Fuel Type", ['Diesel', 'Electric', 'Gas', 'Gasoline', 'Hybrid-Diesel', 'Hybrid-Petrol'])
color = st.selectbox("Color", ['silver', 'blue', 'red', 'black', 'grey', 'brown', 'white', 'green', 'violet', 'orange', 'yellow', 'other'])
transmission_type = st.selectbox("Transmission Type", ['Automatic', 'Manual'])
body_type = st.selectbox("Body Type", ['Sedan', 'SUV', 'Truck'])
drivetrain = st.selectbox("Drivetrain", ['AWD', 'FWD', 'RWD'])
production_year = st.number_input("Production Year", min_value=1900, max_value=2023, step=1)
manufacturer_name = st.selectbox("Manufacturer Name", ['Toyota', 'Ford', 'Tesla'])

#Default values
model_name = 300
# color =
transmission_automatic = 1 if transmission_type == 'Automatic' else 0
transmission_mechanical = 0 if transmission_type == 'Manual' else 0
engine_type_diesel = 1 if engine_type == 'Diesel' else 0
engine_type_electric = 1 if engine_type == 'Electric' else 0
engine_type_gas = 1 if engine_type == 'Gas' else 0
engine_fuel_diesel = 1 if engine_fuel_type == 'Diesel' else 0
engine_fuel_electric = 1 if engine_fuel_type == 'Electric' else 0
engine_fuel_gas = 1 if engine_fuel_type == 'Gas' else 0
engine_fuel_gasoline = 1 if engine_fuel_type == 'Gasoline' else 0
engine_fuel_hybrid_diesel = 1 if engine_fuel_type == 'Hybrid-Diesel' else 0
engine_fuel_hybrid_petrol = 1 if engine_fuel_type == 'Hybrid-Petrol' else 0


def predict_price(input_data):
    input_df = pd.DataFrame(input_data)
    input_df['color'] = le.fit_transform(input_df['color'].values.reshape(-1, 1))
    input_df['color'] = dict(zip(le.classes_, le.transform(le.classes_))).values()
    input_df['body_type'] = le.fit_transform(input_df['color'].values.reshape(-1, 1))
    input_df['body_type'] = dict(zip(le.classes_, le.transform(le.classes_))).values()
    input_df['drivetrain'] = le.fit_transform(input_df['color'].values.reshape(-1, 1))
    input_df['drivetrain'] = dict(zip(le.classes_, le.transform(le.classes_))).values()

    input_data_csv = input_df.to_csv(index=False, header=False)

    # Create a Predictor for the model
    predictor = Predictor(endpoint_name='sagemaker-xgboost-2023-11-29-01-56-43-900',
                          sagemaker_session=sagemaker_session,
                          serializer=CSVSerializer())

    price = predictor.predict(input_data_csv)

    return price

if st.button("Predict Price"):
    input_data = {}
    input_data['manufacturer_name'] = [manufacturer_name]
    input_data['model_name'] = [model_name]
    input_data['color'] = [color]
    input_data['odometer_value'] = [odometer_value]
    input_data['production_year'] = [production_year]
    input_data['transmission_automatic'] = [transmission_automatic]
    input_data['transmission_mechanical'] = [transmission_mechanical]
    input_data['engine_fuel_diesel'] = [engine_fuel_diesel]
    input_data['engine_fuel_electric'] = [engine_fuel_electric]
    input_data['engine_fuel_gas'] = [engine_fuel_gas]
    input_data['engine_fuel_gasoline'] = [engine_fuel_gasoline]
    input_data['engine_fuel_hybrid_diesel'] = [engine_fuel_hybrid_diesel]
    input_data['engine_fuel_hybrid_petrol'] = [engine_fuel_hybrid_petrol]
    input_data['engine_type_diesel'] = [engine_type_diesel]
    input_data['engine_type_electric'] = [engine_type_electric]
    input_data['engine_type_gas'] = [engine_type_gas]
    input_data['body_type'] = [body_type]
    input_data['drivetrain'] = [drivetrain]


    predicted_price = predict_price(input_data)
    st.success(f"Predicted Price: ${predicted_price:,.2f}")



