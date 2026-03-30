"""
 * Monitoring dashboard for real-time flow detection.
 * Created by Shoko on 2026/03/30
 """
import streamlit as st
import pandas as pd
import os
from preprocessor import Preprocessor
from detector import IntrusionDetector

def main():
    st.set_page_config(page_title="LightGBM IDS CIC-2018", layout="wide")
    st.title("Network Intrusion Detection System")

    # Asset loading
    preprocessor = Preprocessor()
    detector = IntrusionDetector()
    
    model_path = os.path.join("models", "lgb_model.joblib")
    if not os.path.exists(model_path):
        st.error("Model artifacts not found. Please run trainer.py first.")
        return

    detector.load(model_path)

    # Input flow analysis
    st.sidebar.header("Connection Flow Parameters")
    
    dst_port = st.sidebar.number_input("Destination Port", min_value=0, value=80)
    protocol = st.sidebar.selectbox("Protocol", [6, 17, 0])
    flow_duration = st.sidebar.number_input("Flow Duration", min_value=0, value=1000)
    tot_fwd_pkts = st.sidebar.number_input("Total Forward Packets", min_value=0, value=1)
    tot_bwd_pkts = st.sidebar.number_input("Total Backward Packets", min_value=0, value=1)

    if st.sidebar.button("Analyze Flow"):
        # Initializing feature vector
        input_data = {
            "Dst Port": [dst_port],
            "Protocol": [protocol],
            "Flow Duration": [flow_duration],
            "Tot Fwd Pkts": [tot_fwd_pkts],
            "Tot Bwd Pkts": [tot_bwd_pkts]
        }
        # Zero-padding for remaining technical features
        for i in range(75):
            input_data[f"feat_{i}"] = [0]
            
        input_df = pd.DataFrame(input_data)
        
        X, _ = preprocessor.preprocess(input_df, is_train=False)
        prob = detector.predict(X)[0]
        
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        col1.metric("Threat Probability", f"{prob:.2%}")
        
        if prob > 0.7:
            col2.error("High risk detected.")
        elif prob > 0.3:
            col2.warning("Suspicious activity.")
        else:
            col2.success("Normal flow.")

        st.bar_chart(pd.DataFrame({
            "Risk Score": [prob],
            "Normal Score": [1 - prob]
        }))

if __name__ == "__main__":
    main()
