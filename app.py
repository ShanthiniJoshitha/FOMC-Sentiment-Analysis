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
#st.sidebar.title("Sentiment Analysis App")
owner_name = "**Dr. Narendra Regmi**"  # Bold the owner's name
st.sidebar.write(owner_name)
st.sidebar.write("Assistant Professor")
st.sidebar.write("Macroeconomics, International Trade, Economic Growth")
st.sidebar.write("Wisconsin University")

# Function to process the input statement
def process_statement(statement):
    lines = statement.split('.')
    sentiments = []

    for line in lines:
        line = line.strip()
        if line:
            try:
                result = sentiment_analyzer(line)
                sentiments.append(result[0]['label'].lower())
            except Exception as e:
                st.warning(f"Error analyzing sentiment for the line: {line}. Error: {e}")
                sentiments.append("neutral")
    
    positive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')
    neutral_count = sentiments.count('neutral')
    total = len(sentiments)

    positive_percentage = (positive_count / total) * 100 if total > 0 else 0
    negative_percentage = (negative_count / total) * 100 if total > 0 else 0
    neutral_percentage = (neutral_count / total) * 100 if total > 0 else 0

    overall_sentiment = max(sentiments, key=sentiments.count) if sentiments else "neutral"

    return {
        "Positive Percentage": positive_percentage,
        "Negative Percentage": negative_percentage,
        "Neutral Percentage": neutral_percentage,
        "Overall Sentiment": overall_sentiment
    }

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
            st.write(f"**Positive Percentage:** {result['Positive Percentage']:.2f}%")
            st.write(f"**Negative Percentage:** {result['Negative Percentage']:.2f}%")
            st.write(f"**Neutral Percentage:** {result['Neutral Percentage']:.2f}%")
            st.write(f"**Overall Sentiment:** {result['Overall Sentiment'].capitalize()}")
        except Exception as e:
            st.error(f"An error occurred during sentiment analysis: {e}")
    else:
        st.warning("Please enter a statement to analyze.")
