# Import necessary libraries
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# Load saved models and vectorizer
log_reg = joblib.load('logistic_regres.pkl')
rf_model = joblib.load('rf_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')  # Pre-trained TF-IDF vectorizer


# Streamlit app UI
st.title("Sentiment Analysis Dashboard")
st.write("Analyze user sentiment from provided text. This application allows you to make predictions using Logistic Regression and Random Forest.")

# Input Section
st.header("Enter your text for sentiment prediction")
user_input = st.text_area("Type your review or feedback below:", height=150)

# Option to choose model
model_choice = st.selectbox(
    "Select a model for prediction:",
    options=["Logistic Regression", "Random Forest"]
)

# Button to predict sentiment
if st.button('Analyze Sentiment'):
    if user_input:
        try:
            # Transform the user input using the saved TF-IDF Vectorizer
            user_input_tfidf = tfidf.transform([user_input])

            # Handle Random Forest feature mismatch by slicing the number of features to expected dimensions
            if model_choice == "Random Forest":
                user_input_tfidf = user_input_tfidf[:, :rf_model.n_features_in_]

            # Perform predictions based on the user's model choice
            if model_choice == "Logistic Regression":
                # Predict and debug what the prediction returns
                raw_prediction = log_reg.predict(user_input_tfidf)  # Debugging here
                st.write(f"Debugging Logistic Regression raw prediction: {raw_prediction}")
                
                # Handle both string and integer predictions safely
                if isinstance(raw_prediction[0], str):  # If string outputs are present
                    sentiment_map = {
                        'negative': 0,
                        'neutral': 1,
                        'positive': 2
                    }
                    mapped_prediction = sentiment_map.get(raw_prediction[0].lower(), -1)  # Map strings to integers
                    if mapped_prediction == -1:
                        st.error("Unexpected model string prediction.")
                    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
                    st.subheader("Prediction Results")
                    st.write(f"### Sentiment Prediction: {sentiment_labels.get(mapped_prediction)}")
                    
                elif isinstance(raw_prediction[0], int):  # If already integers, directly map
                    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
                    st.subheader("Prediction Results")
                    st.write(f"### Sentiment Prediction: {sentiment_labels[raw_prediction[0]]}")
                    
            elif model_choice == "Random Forest":
                # Perform prediction with Random Forest
                raw_prediction = rf_model.predict(user_input_tfidf)[0]  # Extract single prediction
                sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                st.subheader("Prediction Results")
                st.write(f"### Sentiment Prediction: {sentiment_map[raw_prediction]}")
        
        except Exception as e:
            st.error(f"Error during analysis: {e}")
    else:
        st.error("Please enter some text for analysis.")
