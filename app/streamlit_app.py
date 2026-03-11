import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from src.utils import load_model, load_data

# Set Page Config
st.set_page_config(page_title="Vendor Invoice Intelligence", page_icon="📑", layout="wide")

# Custom Styling (Glassmorphism/Modern Look)
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #2e60cf;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 2.5rem;
        background: white;
        border-radius: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Main Title & Description
st.title("📑 Vendor Invoice Intelligence (VII)")
st.caption("Revolutionizing Logistics Audit with Explainable Machine Learning")
st.write("""
Automatic analysis of vendor invoices for **Freight Cost Prediction** and **Risk Detection**. 
Enter the invoice parameters below to receive an instant audit report.
""")

# Sidebar info
with st.sidebar:
    st.image("https://img.icons8.com/bubbles/200/database.png")
    st.info("ℹ️ System Status")
    st.success("🤖 Core: Random Forest")
    st.success("📦 SQL: DuckDB Integrated")
    st.success("⚖️ XAI: SHAP Explainer Enabled")
    st.markdown("---")
    st.write("Built for Enterprise Audits")

# Main Interface Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Prediction Engine", "📊 Model Interpretation", "📂 Dataset Explorer"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📄 Invoice Attributes")
        vendor_id = st.selectbox("Vendor Name", ['Vendor_A', 'Vendor_B', 'Vendor_C', 'Vendor_D', 'Vendor_E'])
        total_amount = st.number_input("Total Invoice Amount (USD)", min_value=1.0, value=2500.0, step=100.0)
        month = st.slider("Month of Invoice", 1, 12, 6)

    with col2:
        st.markdown("### 📦 Item Details")
        quantity = st.number_input("Item Quantity", min_value=1, value=50)
        unit_price = st.number_input("Unit Price (USD)", min_value=0.1, value=45.0)

    if st.button("🚀 Analyze & Predict"):
        try:
            # Load models
            reg_model = load_model('freight_cost_regressor.pkl')
            cls_model = load_model('risk_detector_classifier.pkl')
            
            # Fetch vendor stats (Simplified feature engine logic)
            invoices, _, prices, _ = load_data()
            vendor_avg = invoices[invoices['vendor_id'] == vendor_id]['total_amount'].mean()
            vendor_std = invoices[invoices['vendor_id'] == vendor_id]['total_amount'].std()
            
            # Construct input vector (numerical only)
            input_dict = {
                'total_amount': total_amount,
                'vendor_avg_amt': vendor_avg,
                'vendor_std_amt': vendor_std,
                'invoice_month': month,
                'invoice_dow': 0, # Default Mon
                'quantity': quantity,
                'unit_price': unit_price,
                'item_total_val': quantity * unit_price
            }
            # Add dummy v_ columns
            for v in ['Vendor_A', 'Vendor_B', 'Vendor_C', 'Vendor_D', 'Vendor_E']:
                input_dict[f'v_{v}'] = 1 if vendor_id == v else 0
            
            input_df = pd.DataFrame([input_dict])
            
            # Prediction
            pred_freight = reg_model.predict(input_df)[0]
            
            # Classification
            input_df_cls = input_df.copy()
            input_df_cls['freight_cost'] = pred_freight
            risk_flag = cls_model.predict(input_df_cls)[0]
            risk_proba = cls_model.predict_proba(input_df_cls)[0][1]
            
            # Display
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric("Predicted Freight Cost", f"${pred_freight:.2f}")
                st.progress(min(max(pred_freight / (total_amount * 0.1), 0.0), 1.0), text="Freight vs Total ratio")
            
            with res_col2:
                risk_label = "RISKY 🚩" if risk_flag == 1 else "NORMAL ✅"
                st.metric("Risk Assessment", risk_label)
                st.write(f"Risk Probability: {risk_proba*100:.1f}%")

            if risk_flag == 1:
                st.warning("⚠️ This invoice has been flagged for audit. High probability of financial anomaly.")
            else:
                st.success("👌 This invoice appears to be within normal logistics parameters.")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("💡 Ensure you run 'make train' before running the app.")

with tab2:
    st.header("🔍 Model Interpretation (Explainable AI)")
    st.write("Understanding **why** the model makes certain predictions using SHAP values.")
    
    col_x1, col_x2 = st.columns(2)
    with col_x1:
        st.subheader("Freight Cost Drivers")
        if os.path.exists('outputs/shap_summary_freight_cost_regressor.png'):
            st.image('outputs/shap_summary_freight_cost_regressor.png', caption='Regression Feature Importance (SHAP)')
        else:
            st.warning("SHAP Visualization not found. Run 'make evaluate' to generate.")
            
    with col_x2:
        st.subheader("Invoice Risk Drivers")
        if os.path.exists('outputs/shap_summary_risk_detector_classifier.png'):
            st.image('outputs/shap_summary_risk_detector_classifier.png', caption='Classification Feature Importance (SHAP)')
        else:
            st.info("Run 'make evaluate' to generate risk driver analysis.")

with tab3:
    st.header("📊 Raw Dataset Explorer")
    invoices, _, _, _ = load_data()
    st.write(f"Showing sample of {len(invoices)} invoices.")
    st.dataframe(invoices.head(50), use_container_width=True)
