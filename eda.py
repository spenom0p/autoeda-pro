import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


def generate_eda(df):

    # -----------------------------
    # Dataset Overview
    # -----------------------------
    st.subheader("Dataset Overview")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.write("Missing Values per Column")
    st.dataframe(df.isnull().sum())

    numeric_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns


    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    if len(numeric_cols) > 1:

        st.subheader("Correlation Analysis")

        fig, ax = plt.subplots(figsize=(10,6))

        ax.set_title("Correlation Heatmap (Relationship Between Numeric Variables)")

        sns.heatmap(
            df[numeric_cols].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)


    # -----------------------------
    # Numeric Distributions
    # -----------------------------
    st.subheader("Numeric Feature Distributions")

    for col in numeric_cols:

        fig, ax = plt.subplots()

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        sns.histplot(df[col], kde=True, ax=ax)

        st.pyplot(fig)


    # -----------------------------
    # Categorical Distributions
    # -----------------------------
    if len(cat_cols) > 0:

        st.subheader("Categorical Feature Distributions")

        for col in cat_cols:

            fig, ax = plt.subplots()

            ax.set_title(f"Category Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

            df[col].value_counts().plot(kind="bar", ax=ax)

            st.pyplot(fig)


    # -----------------------------
    # Outlier Detection
    # -----------------------------
    if len(numeric_cols) > 0:

        st.subheader("Outlier Detection")

        for col in numeric_cols:

            fig, ax = plt.subplots()

            ax.set_title(f"Outlier Detection for {col}")
            ax.set_xlabel(col)

            sns.boxplot(x=df[col], ax=ax)

            st.pyplot(fig)