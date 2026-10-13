import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")


# 1. Load the saved model and the encoder 
@st.cache_resource
def load_files():
    return joblib.load("car_price_model.joblib")


saved = load_files()
model = saved["model"]
enc = saved["encoder"]
numeric_features = saved["numeric_features"]
categorical_features = saved["categorical_features"]
clip_low = saved["clip_low"]
clip_high = saved["clip_high"]

options = dict(zip(categorical_features, enc.categories_))

#Page title
st.title("🚗 Used Car Price Predictor")
st.write("Fill in the car details on the left to get an estimated price.")

#Sidebar inputs
st.sidebar.header("Car details")

maker_model = st.sidebar.selectbox("Maker & model", options["Maker_model"])
location = st.sidebar.selectbox("Location", options["Location"])
fuel_type = st.sidebar.selectbox("Fuel type", options["fuel_type"])
transmission = st.sidebar.selectbox("Transmission", options["transmission"])
owner_type = st.sidebar.selectbox("Owner type", options["Owner Type"])

body_options = list(options["body_type"])
body_type = st.sidebar.selectbox("Body type", body_options,
                                 index=body_options.index("Unknown") if "Unknown" in body_options else 0)

age = st.sidebar.slider("Age of car (years)", int(clip_low["Age of car"]), int(clip_high["Age of car"]), 5)
distance = st.sidebar.slider("Distance driven (km)", int(clip_low["Distance"]), int(clip_high["Distance"]),
                             60000, step=1000)
rating = st.sidebar.slider("Vroom audit rating", int(clip_low["Vroom Audit Rating"]),
                           int(clip_high["Vroom Audit Rating"]), 6)
displacement = st.sidebar.number_input("Engine displacement (cc)",
                                       int(clip_low["engine_displacement"]),
                                       int(clip_high["engine_displacement"]), 1600, step=50)
power = st.sidebar.number_input("Engine power (bhp)",
                                int(clip_low["engine_power"]),
                                int(clip_high["engine_power"]), 85, step=5)
doors = st.sidebar.selectbox("Doors", [2, 3, 4, 5], index=2)
seats = st.sidebar.selectbox("Seats", [2, 4, 5, 6, 7], index=2)

car = pd.DataFrame([{
    "Distance": distance,
    "Age of car": age,
    "engine_displacement": displacement,
    "engine_power": power,
    "Vroom Audit Rating": rating,
    "door_count": doors,
    "seat_count": seats,
    "Maker_model": maker_model,
    "Location": location,
    "Owner Type": owner_type,
    "transmission": transmission,
    "fuel_type": fuel_type,
    "body_type": body_type,
}])

car[numeric_features] = car[numeric_features].clip(clip_low, clip_high, axis=1)
car_numbers = car[numeric_features].values
car_text = enc.transform(car[categorical_features])
car_ready = np.hstack([car_numbers, car_text])

price   = np.expm1(model.predict(car_ready))[0] 
price_L = price / 1_00_000                         
low_L   = price_L * 0.95
high_L  = price_L * 1.05

# 7. Show the result
st.subheader("Estimated price")
st.metric("Predicted price", f"₹ {price_L:.2f} L")
st.success(f"Fair price range: ₹ {low_L:.2f} L  to  ₹ {high_L:.2f} L")