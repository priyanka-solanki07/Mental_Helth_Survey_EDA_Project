"""
Mental Health in Tech Survey - Streamlit Dashboard
Run with: streamlit run streamlit_app.py
Place survey.csv in the same folder as this file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set_style("whitegrid")
sns.set_palette("muted")

st.set_page_config(page_title="Mental Health in Tech Survey", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("survey.csv")

    # Clean Age
    df.loc[(df["Age"] < 15) | (df["Age"] > 100), "Age"] = np.nan
    df["Age"] = df["Age"].fillna(df["Age"].median()).astype(int)

    # Clean Gender
    def clean_gender(g):
        g = str(g).strip().lower()
        male_vals = ["male", "m", "man", "cis male", "cis man", "malr", "mal", "maile",
                     "make", "male-ish", "male (cis)", "guy (-ish) ^_^", "msle",
                     "mail", "male leaning androgynous",
                     "ostensibly male, unsure what that really means"]
        female_vals = ["female", "f", "woman", "cis female", "femake", "female (cis)",
                        "female (trans)", "cis-female/femme", "trans woman", "femail"]
        if g in male_vals:
            return "Male"
        elif g in female_vals:
            return "Female"
        return "Other"

    df["Gender"] = df["Gender"].apply(clean_gender)

    # Missing value handling
    df["self_employed"] = df["self_employed"].fillna("No")
    df["work_interfere"] = df["work_interfere"].fillna("Not applicable")
    df["state"] = df["state"].fillna("Not in US")
    df = df.drop(columns=["comments"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    return df


df = load_data()

st.title("Mental Health in Tech Survey (2014) — Dashboard")
st.caption("OSMI survey exploring attitudes toward mental health in the tech workplace.")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")

gender_options = sorted(df["Gender"].unique())
selected_genders = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

country_options = sorted(df["Country"].unique())
top_countries_default = df["Country"].value_counts().head(5).index.tolist()
selected_countries = st.sidebar.multiselect(
    "Country", country_options, default=top_countries_default
)

size_order = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
size_options = [s for s in size_order if s in df["no_employees"].unique()]
selected_sizes = st.sidebar.multiselect("Company Size", size_options, default=size_options)

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

filtered = df[
    df["Gender"].isin(selected_genders)
    & df["Country"].isin(selected_countries)
    & df["no_employees"].isin(selected_sizes)
    & df["Age"].between(age_range[0], age_range[1])
]

st.sidebar.markdown(f"**{len(filtered)} of {len(df)} respondents match filters**")

if filtered.empty:
    st.warning("No data matches the selected filters. Adjust filters in the sidebar.")
    st.stop()

# ---------------- KPI row ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Respondents", len(filtered))
col2.metric("Sought Treatment", f"{(filtered['treatment'] == 'Yes').mean() * 100:.1f}%")
col3.metric("Family History", f"{(filtered['family_history'] == 'Yes').mean() * 100:.1f}%")
col4.metric("Has Benefits", f"{(filtered['benefits'] == 'Yes').mean() * 100:.1f}%")

st.divider()

# ---------------- Row 1: distributions ----------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Treatment by Gender")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=filtered, x="Gender", hue="treatment", ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)

with c2:
    st.subheader("Treatment by Family History")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=filtered, x="family_history", hue="treatment", ax=ax)
    ax.set_xlabel("Family History")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# ---------------- Row 2: benefits & work interference ----------------
c3, c4 = st.columns(2)

with c3:
    st.subheader("Treatment by Benefits Availability")
    fig, ax = plt.subplots(figsize=(6, 4))
    order = filtered["benefits"].value_counts().index
    sns.countplot(data=filtered, x="benefits", hue="treatment", order=order, ax=ax)
    ax.set_xlabel("Benefits")
    st.pyplot(fig)

with c4:
    st.subheader("Treatment by Work Interference")
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["Never", "Rarely", "Sometimes", "Often", "Not applicable"]
    order = [o for o in order if o in filtered["work_interfere"].unique()]
    sns.countplot(data=filtered, x="work_interfere", hue="treatment", order=order, ax=ax)
    ax.set_xlabel("Work Interference")
    plt.xticks(rotation=15)
    st.pyplot(fig)

# ---------------- Row 3: company size & age ----------------
c5, c6 = st.columns(2)

with c5:
    st.subheader("Treatment by Company Size")
    fig, ax = plt.subplots(figsize=(6, 4))
    order = [s for s in size_order if s in filtered["no_employees"].unique()]
    sns.countplot(data=filtered, x="no_employees", hue="treatment", order=order, ax=ax)
    ax.set_xlabel("Number of Employees")
    plt.xticks(rotation=20)
    st.pyplot(fig)

with c6:
    st.subheader("Age Distribution by Treatment Status")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=filtered, x="treatment", y="Age", ax=ax)
    ax.set_xlabel("Sought Treatment")
    st.pyplot(fig)

# ---------------- Row 4: country treatment rate ----------------
st.subheader("Treatment Rate by Country (Selected)")
country_rate = (
    filtered.groupby("Country")["treatment"]
    .apply(lambda x: (x == "Yes").mean())
    .sort_values(ascending=False)
)
fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * len(country_rate))))
sns.barplot(x=country_rate.values, y=country_rate.index, ax=ax)
ax.set_xlabel("Proportion Who Sought Treatment")
ax.set_ylabel("Country")
st.pyplot(fig)

# ---------------- Raw data ----------------
with st.expander("View filtered raw data"):
    st.dataframe(filtered)
