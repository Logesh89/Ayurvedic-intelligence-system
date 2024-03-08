import streamlit as st
import pandas as pd
from homepage import home
from selectSymptoms import SymptomSelection
from sym import mainFunc
# data = pd.read_csv("/path")

st.title("Ayurvedic Intelligent System for Personalized Herbal Recommendations")

nav = st.sidebar.radio("Navigation",["Home","Select Symptoms"])
if nav == "Home":
    home()
elif nav == "Select Symptoms":
  # SymptomSelection()
  # sym()
  mainFunc()