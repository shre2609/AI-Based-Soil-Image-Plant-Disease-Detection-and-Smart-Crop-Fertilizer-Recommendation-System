import streamlit as st
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import joblib
import requests
from datetime import datetime
import google.generativeai as genai
import os

# Gemini API Setup
genai.configure(api_key="AIzaSyChtDxrHJ0ALChw5GLLnK9Qemyb8uV1kL0")

# Load Crop & Fertilizer Models
crop_model = joblib.load("crop_recommendation_model_xgb.pkl")
crop_scaler = joblib.load("scaler.pkl")
crop_label_enc = joblib.load("label_encoder.pkl")
crop_features = joblib.load("feature_columns.pkl")

fert_model = joblib.load("fertilizer_model.pkl")
fert_scaler = joblib.load("fertilizer_scaler.pkl")
fert_label_enc = joblib.load("fertilizer_encoder.pkl")

# Load Soil Image Model
@st.cache_resource
def load_soil_model():
    model = tf.keras.models.load_model("soil_model.keras")
    return model

soil_model = load_soil_model()
soil_classes = ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"]

# Helper Functions
def get_location():
    try:
        res = requests.get("https://ipinfo.io/json")
        data = res.json()
        lat, lon = map(float, data["loc"].split(","))
        city = data.get("city", "Unknown")
        region = data.get("region", "")
        country = data.get("country", "")
        return lat, lon, f"{city}, {region}, {country}"
    except:
        return 18.52, 73.85, "Pune, India"

def fetch_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
    response = requests.get(url)
    data = response.json()
    temperature = data["current"]["temperature_2m"]
    humidity = data["current"]["relative_humidity_2m"]
    rainfall = data["current"]["precipitation"]
    return temperature, humidity, rainfall

def get_season():
    month = datetime.now().month
    if 6 <= month <= 10:
        return "Kharif"
    elif month in [11,12,1,2,3]:
        return "Rabi"
    else:
        return "Zaid"

def make_card(title, subtitle, color="#4CAF50"):
    return f"""
    <div style="padding:15px;margin:10px;background:{color};
                border-radius:15px;box-shadow:0 4px 10px rgba(0,0,0,0.2);
                color:white;font-weight:bold;text-align:center;">
        <h3 style="margin:0;">{title}</h3>
        <p style="margin:0;font-size:16px;">{subtitle}</p>
    </div>
    """

# Streamlit UI Setup
st.set_page_config(page_title="🌱 AI Crop & Fertilizer", layout="centered")
st.markdown("""
<style>
.stTabs [role="tablist"] button {font-size:16px;font-weight:600;border-radius:10px;padding:8px 20px;}
.stTabs [role="tablist"] button[aria-selected="true"] {background:#4CAF50;color:white;}
</style>
""", unsafe_allow_html=True)

st.title("🌱 AI-based Soil, Crop & Fertilizer Recommendation")

tab_options = ["🖼 Soil Classifier","🌾 Crop Recommendation","🧪 Fertilizer Suggestion",
               "🤖 Gemini AI Summary","🏛 Government Schemes"]

tabs = st.tabs(tab_options)

# Tab 0: Soil Classifier
with tabs[0]:
    st.header("🖼 Soil Type Classifier")
    st.write("Upload a soil image and the model will predict its type.")
    uploaded_file = st.file_uploader("Choose a soil image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        img = load_img(uploaded_file, target_size=(224,224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        pred = soil_model.predict(arr)
        class_idx = np.argmax(pred)
        confidence = pred[0][class_idx]*100
        st.success(f"🔍 Predicted Soil Type: **{soil_classes[class_idx]}** ({confidence:.2f}%)")

# Tab 1: Crop Recommendation
with tabs[1]:
    st.header("🌾 Smart Crop Recommendation")
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200, 50)
        P = st.number_input("Phosphorus (P)", 0, 200, 40)
    with col2:
        K = st.number_input("Potassium (K)", 0, 200, 40)
        ph = st.number_input("pH", 0.0, 14.0, 6.5, step=0.1)

    if st.button("🔍 Recommend Crop"):
        try:
            lat, lon, location_name = get_location()
            temperature, humidity, rainfall = fetch_weather(lat, lon)
            season = get_season()

            crop_input = pd.DataFrame(columns=crop_features)
            crop_input.loc[0] = 0
            crop_input.loc[0, ["N","P","K","temperature","humidity","ph","rainfall"]] = [N,P,K,temperature,humidity,ph,rainfall]
            season_col = f"season_{season}"
            if season_col in crop_input.columns:
                crop_input.loc[0, season_col] = 1

            crop_scaled = crop_scaler.transform(crop_input)
            probs = crop_model.predict_proba(crop_scaled)[0]
            top_idx = np.argmax(probs)
            top_crop = crop_label_enc.inverse_transform([top_idx])[0]
            top_prob = probs[top_idx]*100

            st.session_state["top_crop"] = top_crop
            st.session_state["soil_features"] = {
                "N": N,"P": P,"K": K,"ph": ph,
                "temperature": temperature,"humidity": humidity,
                "rainfall": rainfall,"season": season,"location": location_name
            }

            st.markdown(make_card("📍 Location", location_name, "#2E7D32"), unsafe_allow_html=True)
            st.markdown(make_card("🗓 Season", season, "#43A047"), unsafe_allow_html=True)
            colA, colB, colC = st.columns(3)
            colA.markdown(make_card("🌡 Temp", f"{temperature:.1f} °C", "#388E3C"), unsafe_allow_html=True)
            colB.markdown(make_card("💧 Humidity", f"{humidity:.1f} %", "#388E3C"), unsafe_allow_html=True)
            colC.markdown(make_card("🌧 Rainfall", f"{rainfall:.1f} mm", "#388E3C"), unsafe_allow_html=True)

            st.subheader("🌱 Recommended Crop")
            st.markdown(make_card(top_crop, f"{top_prob:.2f} %", "#689F38"), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error fetching weather/crop: {e}")

# Tab 2: Fertilizer Suggestion
with tabs[2]:
    st.header("🧪 Fertilizer Suggestions")
    if "top_crop" not in st.session_state:
        st.warning("⚠ Please run the Crop Recommendation first.")
    else:
        soil_features = st.session_state["soil_features"]
        crop_name = st.session_state["top_crop"]

        fert_input = pd.DataFrame([{
            "Nitrogen N": soil_features["N"],"Phosphorus P": soil_features["P"],"Potassium K": soil_features["K"],
            "temperature": soil_features["temperature"],"humidity": soil_features["humidity"],
            "ph": soil_features["ph"],"rainfall": soil_features["rainfall"],
            "season": soil_features["season"],"crop_type": crop_name
        }])
        fert_input["season"] = fert_input["season"].astype("category").cat.codes
        fert_input["crop_type"] = fert_input["crop_type"].astype("category").cat.codes
        fert_scaled = fert_scaler.transform(fert_input)
        fert_pred = fert_model.predict(fert_scaled)[0]
        fert_category = fert_label_enc.inverse_transform([fert_pred])[0]

        st.subheader(f"🌾 Crop: {crop_name}")
        st.markdown(make_card("🧪 Fertilizer Category", fert_category, "#00796B"), unsafe_allow_html=True)

        gemini_prompt = f"Suggest 3-5 actual fertilizers with NPK for category: {fert_category}"
        try:
            gemini_model = genai.GenerativeModel("gemini-2.5-pro")
            response = gemini_model.generate_content(gemini_prompt)
            fert_suggestions = getattr(response, "content", None) or getattr(response, "text", "No response")
            fert_suggestions = fert_suggestions.strip()
        except Exception as e:
            fert_suggestions = f"Gemini API Error: {e}"

        st.markdown(make_card("💡 Suggested Fertilizers", fert_suggestions, "#009688"), unsafe_allow_html=True)
        st.session_state["fert_result"] = {"crop": crop_name,"category": fert_category,"suggestions": fert_suggestions}

# Tab 3: Gemini AI Summary
with tabs[3]:
    st.header("🤖 Gemini AI Summary")
    available_models = ["gemini-2.5-pro","gemini-2.5-flash","gemini-2.0-flash"]
    selected_model = st.selectbox("Select Gemini Model", available_models)
    gemini_model = genai.GenerativeModel(selected_model)

    soil_features = st.session_state.get("soil_features")
    top_crop = st.session_state.get("top_crop")
    fert_result = st.session_state.get("fert_result")

    if soil_features and top_crop and fert_result:
        combined_info = f"""
Location: {soil_features['location']}
Soil: N={soil_features['N']}, P={soil_features['P']}, K={soil_features['K']}, pH={soil_features['ph']}
Weather: Temp={soil_features['temperature']}°C, Humidity={soil_features['humidity']}%, Rainfall={soil_features['rainfall']}mm
Season: {soil_features['season']}
Crop: {top_crop}
Fertilizer: {fert_result}
"""
        try:
            response = gemini_model.generate_content(combined_info)
            summary_text = getattr(response, "content", None) or getattr(response, "text", "No response")
            st.session_state["gemini_summary"] = summary_text.strip()
        except Exception as e:
            st.error(f"Gemini API Error: {e}")

        if "gemini_summary" in st.session_state:
            st.subheader("📋 Gemini’s Explanation")
            st.write(st.session_state["gemini_summary"])
    else:
        st.info("⚠ Run Crop & Fertilizer first.")

# Tab 4: Government Schemes
with tabs[4]:
    st.header("🏛 Government Schemes")
    
    # Example: many schemes to test scrolling
    schemes = [
        {"name":"PM Fasal Bima Yojana","desc":"Crop insurance","url":"https://pmfby.gov.in"},
        {"name":"PM-Kisan","desc":"Direct income support","url":"https://pmkisan.gov.in"},
        {"name":"eNAM","desc":"Online agri market","url":"https://enam.gov.in"},
        {"name":"Kisan Credit Card","desc":"Farm credit","url":"https://www.rbi.org.in/Scripts/FAQView.aspx?Id=92"},
        {"name":"National Agriculture Market","desc":"Market reforms","url":"https://www.namonline.in/"},
        {"name":"Rashtriya Krishi Vikas Yojana","desc":"Agricultural development","url":"https://rkvy.nic.in/"},
        {"name":"Pradhan Mantri Krishi Sinchayee Yojana","desc":"Irrigation support","url":"https://pmksy.gov.in/"},
        {"name":"Soil Health Card Scheme","desc":"Soil testing & improvement","url":"https://soilhealth.dac.gov.in/"},
    ]

    for s in schemes:
        st.markdown(f"[{s['name']}]({s['url']}) - {s['desc']}")