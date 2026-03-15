# app.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
import re
import unicodedata

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("ChatGroq_API_KEY")

# --- Initialize LLM ---
llm = ChatGroq(
    temperature=0,
    api_key=api_key,
    model="llama-3.3-70b-versatile"
)

# --- Prompt Templates ---
categorize_prompt = PromptTemplate(
    input_variables=["description"],
    template="Categorize this transaction: '{description}'. Respond with one word category like Food, Transport, Rent, Shopping, Subscription, Other."
)
categorizer_chain = LLMChain(llm=llm, prompt=categorize_prompt)

summary_prompt = PromptTemplate(
    input_variables=["spending_data"],
    template="""
    You are a financial assistant.
    Given this spending summary: {spending_data}
    Write a short report (3-4 sentences) about spending habits and suggest 2 improvements.
    """
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# --- Functions ---
def load_transactions(uploaded_file):
    """
    Load CSV or Excel transactions from a Streamlit uploaded file.
    """
    # Check file extension using the name
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]
    df.rename(columns={
        'description': 'desc',
        'amount': 'amount',
        'date': 'date',
        'type': 'type'
    }, inplace=True)

    # Type conversions
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop incomplete rows
    df.dropna(subset=['desc', 'amount', 'date'], inplace=True)
    df.sort_values('date', inplace=True)

    return df


def categorize_transactions(df):
    df['category'] = df['desc'].apply(lambda x: categorizer_chain.run(description=x))
    return df

def detect_suspicious(df):
    suspicious = []
    duplicates = df[df.duplicated(['amount', 'date'], keep=False)]
    if not duplicates.empty:
        suspicious.append("Duplicate transactions found.")

    debit_df = df[df['type'].str.lower() == 'debit']
    mean_debit = debit_df['amount'].mean()
    large_spend = debit_df[debit_df['amount'] > mean_debit * 3]
    if not large_spend.empty:
        details = large_spend[['date', 'desc', 'amount']].to_dict(orient='records')
        details_str = "\n".join([f"- {d['date']} | {d['desc']} | ${d['amount']}" for d in details])
        suspicious.append(f"Unusually large debit transactions detected:\n{details_str}")
    return suspicious

def clean_llm_output(text: str) -> str:
    """
    Fix common formatting issues in LLM output, like character-separated text.
    """
    # collapse whitespace
    text = ' '.join(text.split())

    # heuristic: if too many single-letter tokens, collapse them
    tokens = text.split()
    single_count = sum(1 for t in tokens if len(t) == 1)
    if single_count > len(tokens) * 0.35:
        # merge separated letters
        prev = None
        while prev != text:
            prev = text
            text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', text)
        # fix punctuation spacing
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        text = ' '.join(text.split())

    return text.strip()


def clean_ai_text(text):
    # Normalize Unicode text (handles weird spacing chars)
    text = unicodedata.normalize("NFKC", text)
    
    # Remove all non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    
    # Replace multiple newlines or spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Add space after punctuation if missing
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    
    # Fix missing spaces between words (in case of CamelCase-like merges)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    return text.strip()


def summarize_finances(df):
    summary_dict = df.groupby('category')['amount'].sum().to_dict()
    summary_text = "\n".join([f"{cat}: ${amt:.2f}" for cat, amt in summary_dict.items()])

    raw_response = summary_chain.run(spending_data=summary_text)
    cleaned_response = clean_ai_text(raw_response)

    return cleaned_response





def plot_spending(df):
    category_totals = df.groupby('category')['amount'].sum()
    fig, ax = plt.subplots(figsize=(6,6))
    category_totals.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    ax.set_title("Spending by Category")
    st.pyplot(fig)



# --- Streamlit App ---
st.title(" Financial Transaction Analyzer")


uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
if uploaded_file:
    df = load_transactions(uploaded_file)
    st.subheader("Raw Transactions")
    st.dataframe(df.head())

    # Categorize
    with st.spinner("Categorizing transactions..."):
        df = categorize_transactions(df)
    st.subheader("Transactions with Categories")
    st.dataframe(df.head())

    # Detect suspicious
    issues = detect_suspicious(df)
    if issues:
        st.subheader("🚨 Issues Found")
        for issue in issues:
            st.write(issue)
    else:
        st.write("No suspicious transactions detected.")

    # Summary
    ai_summary = summarize_finances(df)
    st.markdown(f"### 📊 AI Summary\n\n{ai_summary}")

    # Plot
    st.subheader("Spending by Category")
    plot_spending(df)
