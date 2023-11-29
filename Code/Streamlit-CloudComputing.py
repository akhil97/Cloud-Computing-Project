import streamlit as st
import pandas as pd
from sagemaker import image_uris, session
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel

st.set_page_config(page_title="Team 3 Cloud Computing Project", page_icon=":tada:", layout="wide")

st.subheader("This is project to predict selling price of used cars")

#Inputs from user
odometer_value = st.number_input("Odometer Value", min_value=0)
engine_type = st.selectbox("Engine Type", ['Gas', 'Diesel', 'Electric'])
body_type = st.selectbox("Body Type", ['Sedan', 'SUV', 'Truck'])
drivetrain = st.selectbox("Drivetrain", ['AWD', 'FWD', 'RWD'])
production_year = st.number_input("Production Year", min_value=1900, max_value=2023, step=1)
manufacturer_name = st.selectbox("Manufacturer Name", ['Toyota', 'Ford', 'Tesla'])

#Default values
model_name = 300
color = 0
transmission_automatic = 1
transmission_mechanical = 0
engine_type_diesel = 1 if engine_type == 'Diesel' else 0
engine_type_electric = 1 if engine_type == 'Electric' else 0
engine_type_gas = 1 if engine_type == 'Gas' else 0
engine_fuel_diesel = engine_type_diesel
engine_fuel_electric = engine_type_electric
engine_fuel_gas = engine_type_gas
engine_fuel_gasoline = 0
engine_fuel_hybrid_diesel = 0
engine_fuel_hybrid_petrol = 0


def predict_price(input_data):
    input_df = pd.DataFrame(input_data)
    xgb_image = image_uris.retrieve("xgboost", session.Session().boto_region_name, repo_version="latest")
    xgb_model = Model(model_data="s3://myccprojectbucket/output/sagemaker-xgboost-2023-11-29-01-52-54-334/output//model.tar.gz", image_uri=xgb_image)

    sagemaker_model = "sagemaker-xgboost-2023-11-29-01-56-43-900"

    sm_model = PipelineModel(name=sagemaker_model, role=data_scientist, models=[xgb_model])
    price = sm_model.predict(input_df).decode('utf-8')
    return price




if st.button("Predict Price"):
    input_data = [manufacturer_name, model_name, color, odometer_value, production_year, transmission_automatic,
                  transmission_mechanical, engine_fuel_diesel, engine_fuel_electric, engine_fuel_gas, engine_fuel_gasoline,
                  engine_fuel_hybrid_diesel, engine_fuel_hybrid_petrol, engine_type_diesel, engine_type_electric,
                  engine_fuel_gasoline, body_type, drivetrain]
    
    predicted_price = predict_price(None, input_data)
    st.success(f"Predicted Price: ${predicted_price:,.2f}")



