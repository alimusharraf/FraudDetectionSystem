# TASK 6 — Streamlit Fraud Operations Dashboard

# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt


# Configure Streamlit page
st.set_page_config(page_title='Fraud Detection Dashboard', layout='wide')


# Dashboard title
st.title('Real-Time Fraud Detection System')


# Load saved model
model = joblib.load('dashboard/model.pkl')


# Load scaler
scaler = joblib.load('dashboard/scaler.pkl')


# Load processed dataset
df = pd.read_csv('data/final_data_sample.csv')


# Sidebar navigation menu
# Configure Streamlit page

st.set_page_config(
    page_title='Fraud Detection Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Sidebar title
st.sidebar.markdown("## Fraud Dashboard")

# Default page
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

# Navigation buttons
if st.sidebar.button('Overview'):
    st.session_state.page = 'Overview'

if st.sidebar.button('Transaction Explorer'):
    st.session_state.page = 'Transaction Explorer'

if st.sidebar.button('SHAP Explainer'):
    st.session_state.page = 'SHAP Explainer'

# Current selected page
page = st.session_state.page

# Overview page
if page == 'Overview':

    st.header('Fraud Overview Dashboard')


    # Calculate metrics
    total_transactions = len(df)

    total_fraud = df['isFraud'].sum()

    fraud_rate = (total_fraud / total_transactions) * 100

    avg_fraud_amt = df[df['isFraud'] == 1]['TransactionAmt'].mean()


    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric('Total Transactions', total_transactions)

    col2.metric('Fraud Transactions', int(total_fraud))

    col3.metric('Fraud Rate (%)', f'{fraud_rate:.2f}')

    col4.metric('Average Fraud Amount', f'{avg_fraud_amt:.2f}')


    # Risk tier distribution chart
    st.subheader('Risk Tier Distribution')

    risk_counts = df['RiskTier'].value_counts()

    fig = px.pie(

        values=risk_counts.values,

        names=risk_counts.index,

        hole=0.5,

        title='Risk Tier Distribution'
    )

    st.plotly_chart(fig, use_container_width=True)


    # Fraud rate by hour chart
    st.subheader('Fraud Rate by Hour')

    hour_fraud = df.groupby('HourOfDay')['isFraud'].mean().reset_index()

    fig2 = px.line(

        hour_fraud,

        x='HourOfDay',

        y='isFraud',

        title='Fraud Rate by Hour'
    )

    st.plotly_chart(fig2, use_container_width=True)


# Transaction explorer page
elif page == 'Transaction Explorer':

    st.header('Transaction Explorer')


    # Transaction selection

    selected_id = st.selectbox(

        'Select or Search TransactionID',

        df['TransactionID'].sample(1000)
    )

    # Load button
    if st.button('Load Transaction'):
        # Filter selected transaction
        filtered_df = df[df['TransactionID'] == selected_id]
        
        #Display transaction
        st.subheader('Transaction Details')
        st.dataframe(filtered_df)


        # Fraud probability display
        if not filtered_df.empty:

            probability = filtered_df['FraudProbability'].values[0]

            risk_tier = filtered_df['RiskTier'].values[0]

            st.metric('Fraud Probability', f'{probability:.4f}')

            st.metric('Risk Tier', risk_tier)


# SHAP explainer page
elif page == 'SHAP Explainer':

    st.header('SHAP Explainability')


    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)


    # SHAP transaction selection
    shap_transaction = st.selectbox(
        'Select or Search TransactionID for SHAP Analysis',
        df['TransactionID'].sample(1000)
    )

    # Load SHAP analysis button
    if st.button('Generate SHAP Analysis'):

        # Filter selected transaction
        selected_row = df[df['TransactionID'] == shap_transaction]

        if not selected_row.empty:
            # Remove non-feature columns
            feature_data = selected_row.drop(
                ['isFraud', 'FraudProbability', 'RiskTier'],
                axis=1,
                errors='ignore'
            )


            # Scale selected transaction
            scaled_data = scaler.transform(feature_data)

            scaled_data = pd.DataFrame(

                scaled_data,

                columns=feature_data.columns
            )

            # Generate SHAP values
            shap_values = explainer.shap_values(scaled_data)

            # Predict probability
            prediction = model.predict_proba(scaled_data)[:,1][0]

            # SHAP waterfall plot
            st.subheader('SHAP Waterfall Plot')

            fig, ax = plt.subplots(figsize=(10,6))

            shap.plots.waterfall(

                shap.Explanation(

                    values=shap_values[0],

                    base_values=explainer.expected_value,

                    data=scaled_data.iloc[0],

                    feature_names=scaled_data.columns
                ),

                show=False
            )

            st.pyplot(fig)


            # Plain-English explanation
            st.subheader('Plain-English Explanation')


            if prediction >= 0.75:

                st.error(

                    'This transaction shows strong fraud indicators including abnormal transaction behavior and suspicious feature patterns.'
                )

            elif prediction >= 0.40:

                st.warning(

                    'This transaction contains both legitimate and suspicious characteristics and should be reviewed further.'
                )

            else:

                st.success(

                    'This transaction appears consistent with normal transaction behavior patterns.'
                )