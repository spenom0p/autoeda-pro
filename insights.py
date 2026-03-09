import streamlit as st

def generate_insights(df):

    st.subheader("Dataset Insights")

    st.write("Total Rows:", df.shape[0])
    st.write("Total Columns:", df.shape[1])

    st.write("Total Missing Values:", df.isnull().sum().sum())

    st.write("Duplicate Rows:", df.duplicated().sum())