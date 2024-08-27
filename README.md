import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


@st.cache(allow_output_mutation=True)
def load_data(file):
    """Load data from a CSV file."""
    return pd.read_csv(file)


def train_model(data):
    """Train a logistic regression model on the given data."""
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)

    X = data.drop(columns="Class")
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    save_model(model, 'fraud_detection_model.pkl')  # Save the model

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    return model, train_acc, test_acc


def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)


def load_model(filename):
    """Load the trained model from a file."""
    return joblib.load(filename)


st.title("Credit Card Fraud Detection")

file = st.file_uploader("Upload your credit card transaction CSV file:")
if file is not None:
    data = load_data(file)
    st.write("Data shape:", data.shape)

    # Check if model file exists
    try:
        model = load_model('fraud_detection_model.pkl')
    except FileNotFoundError:
        model, train_acc, test_acc = train_model(data)
        st.write("Training accuracy:", train_acc)
        st.write("Test accuracy:", test_acc)
    else:
        st.write("Model loaded successfully")

    st.subheader("Check a transaction")

    feature_names = data.drop(columns="Class").columns.tolist()

    # Create input fields for each feature
    features = {}
    for feature in feature_names:
        # Create input widgets for each feature
        features[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

    if st.button("Predict"):
        try:
            # Convert input values to a DataFrame
            transaction_df = pd.DataFrame([features])
            prediction = model.predict(transaction_df)
            st.write("Prediction:", "Fraudulent" if prediction[0] == 1 else "Legitimate")
        except Exception as e:
            st.error(f"Error processing input: {e}")
