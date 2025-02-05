from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import streamlit as st
import pandas as pd

# Get Hugging Face token from Streamlit secrets
hf_token = st.secrets.get("HF_AUTH_TOKEN")

if not hf_token:
    st.error("Hugging Face token is missing! Please add it to Streamlit secrets.")
    st.stop()

# Define the model name
model_name = "yiyanghkust/finbert-tone"

# Load the tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=hf_token)
except Exception as e:
    st.warning(f"Fast tokenizer failed to load: {e}. Falling back to the slow tokenizer.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, use_auth_token=hf_token)
    except Exception as e:
        st.error(f"Error loading slow tokenizer: {e}")
        st.stop()

# Load the model
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load Excel file from backend
backend_excel_path = "reports-in.xlsx"
try:
    df = pd.read_excel(backend_excel_path)
    if "Date" not in df.columns or "Statement" not in df.columns:
        st.error("Excel file must contain 'Date' and 'Statement' columns.")
        st.stop()
    
    # Extract year from Date column and convert to integer
    df["Year"] = pd.to_datetime(df["Date"], errors='coerce').dt.year.astype('Int64')
except Exception as e:
    st.error(f"Error loading Excel file: {e}")
    st.stop()

# Streamlit Sidebar - subtle ownership mention
owner_name = "**Dr. Narendra Regmi**"  # Bold the owner's name
st.sidebar.write(owner_name)
st.sidebar.write("Assistant Professor")
st.sidebar.write("Macroeconomics, International Trade, Economic Growth")
st.sidebar.write("Wisconsin University")

# Function to process the input statement
def process_statement(statement):
    # Analyze the sentiment of the full statement rather than individual sentences
    try:
        result = sentiment_analyzer(statement)

        # Debugging: Output raw result for inspection
        st.write("Raw Model Output: ", result)

        sentiment = result[0]['label'].lower() if result else "neutral"
        
        # Counting sentiment types
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        if sentiment == 'positive':
            positive_count += 1
        elif sentiment == 'negative':
            negative_count += 1
        else:
            neutral_count += 1

        total = positive_count + negative_count + neutral_count  # Total count of sentiments

        positive_percentage = (positive_count / total) * 100 if total > 0 else 0
        negative_percentage = (negative_count / total) * 100 if total > 0 else 0

        # Compare positive and negative percentages for overall sentiment
        if positive_percentage > negative_percentage:
            overall_sentiment = "positive"
        elif negative_percentage > positive_percentage:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        return {
            "Positive Percentage": positive_percentage,
            "Negative Percentage": negative_percentage,
            "Overall Sentiment": overall_sentiment
        }

    except Exception as e:
        st.warning(f"Error during sentiment analysis: {e}")
        return {"Positive Percentage": 0, "Negative Percentage": 0, "Overall Sentiment": "neutral"}

# Streamlit App UI
st.title("FOMC Sentiment Analysis")
st.write("Analyze the sentiment of FOMC statements using a fine-tuned financial model.")

# Navigation panel
option = st.radio("Select Input Method:", ("Excel", "Direct"))

if option == "Excel":
    years = df["Year"].dropna().unique()
    selected_year = st.selectbox("Select Year:", sorted(years, reverse=True))
    
    statements = df[df["Year"] == selected_year]["Statement"].tolist()
    selected_statement = st.selectbox("Select Statement:", statements)
    
    user_input = st.text_area("Selected Statement:", selected_statement)
else:
    user_input = st.text_area("Enter your statement below:", placeholder="Type here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        try:
            result = process_statement(user_input)
            st.subheader("Sentiment Analysis Results:")
            st.write(f"In whole, the above statement has **{result['Positive Percentage']:.2f}%** positive and **{result['Negative Percentage']:.2f}%** negative sentiment.")
            st.write(f"The overall sentiment of the statement is **{result['Overall Sentiment'].capitalize()}**.")
        except Exception as e:
            st.error(f"An error occurred during sentiment analysis: {e}")
    else:
        st.warning("Please enter a statement to analyze.")
