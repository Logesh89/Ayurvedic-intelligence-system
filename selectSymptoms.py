import streamlit as st
from data.symptoms import symptoms
from result import result
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf

data = {}

def SymptomSelection():
    # Dropdown menus for symptoms
    symptom1 = st.selectbox("Symptom 1", ["Select a Symptom"]+symptoms)
    symptom2 = st.selectbox("Symptom 2", ["Select a Symptom"]+symptoms)
    symptom3 = st.selectbox("Symptom 3", ["Select a Symptom"]+symptoms)
    symptom4 = st.selectbox("Symptom 4", ["Select a Symptom"]+symptoms)
    symptom5 = st.selectbox("Symptom 5", ["Select a Symptom"]+symptoms)

    # Load data
    tr = pd.read_csv("Testing.csv")
    tr.replace({'prognosis': {'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
     'Migraine':11,'Cervical spondylosis':12,
     'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
     'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
      'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
      'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
       '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
        'Impetigo':40}}, inplace=True)

    X_test = tr[symptoms]
    y_test = tr[["prognosis"]]
    np.ravel(y_test)

    df = pd.read_csv("Training.csv")
    df.replace({'prognosis': {'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
     'Migraine':11,'Cervical spondylosis':12,
     'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
     'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
      'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
      'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
       '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
        'Impetigo':40}}, inplace=True)

    X = df[symptoms]
    y = df[["prognosis"]]
    np.ravel(y)

    def message():
        if symptom1 == "None" and symptom2 == "None" and symptom3 == "None" and symptom4 == "None" and symptom5 == "None":
            print("No output")
        else:
            # XGBoost
            XGBoost()

    def XGBoost():
        model = xgb.XGBClassifier()
        model.fit(X, np.ravel(y))
        
        psymptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]
        l2 = [1 if symptom in psymptoms else 0 for symptom in symptoms]

        inputtest = [l2]
        predicted = model.predict(inputtest)[0]

        data = {'disease': predicted}

    # ANN Functions
    def ANNModel():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(len(symptoms),)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len('disease'), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def TrainANN(X_train, y_train):
        model = ANNModel()
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        return model

    def ANN_Prediction(model, input_data):
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        return 'disease'[predicted_class]

    if st.button("Submit"):
        XGBoost()  # Use XGBoost
        # For ANN:
        # model = TrainANN(X, y)  # Train ANN using training data
        # predicted_disease = ANN_Prediction(model, inputtest)  # Predict disease using trained ANN
        # data = {'disease': predicted_disease}
        result(data)
