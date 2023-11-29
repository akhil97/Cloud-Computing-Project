import streamlit as st
# Import model here


st.set_page_config(page_title="Team 3 Cloud Computing Project", page_icon=":tada:", layout="wide")

st.subheader("This is project to predict selling price of used cars")


#model = model_name(data)


odometer_value = st.number_input("Odometer Value", min_value=0)
engine_type = st.selectbox("Engine Type", ['Gas', 'Diesel', 'Electric'])
body_type = st.selectbox("Body Type", ['Sedan', 'SUV', 'Truck'])
drivetrain = st.selectbox("Drivetrain", ['AWD', 'FWD', 'RWD'])
production_year = st.number_input("Production Year", min_value=1900, max_value=2023, step=1)
manufacturer_name = st.selectbox("Manufacturer Name", ['Toyota', 'Ford', 'Tesla'])

if st.button("Predict Price"):
    input_data = [odometer_value, engine_type, body_type, drivetrain, production_year, manufacturer_name]

    predicted_price = model.predict([input_data])[0]
    st.success(f"Predicted Price: ${predicted_price:,.2f}")
