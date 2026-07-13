import streamlit as st
import pandas as pd
from utils import model_helper 

st.header('Claim Fraud Preditcion')

file = st.file_uploader("Upload data yang ingin dipredisksi (csv)", type=".csv")

if file:
    df_file = pd.read_csv(file)
    st.dataframe(df_file)
    predict = st.button('Predict !')

    if predict:
        result = model_helper.predict(df_file)
        predict_proba = result['prob']
        predict_class = result['class']

        result_df = df_file[['claim_number']].copy()
    
        result_df['prediction'] = predict_class
        result_df['probability'] = predict_proba

        st.success('Prediksi Berhasil')
        st.dataframe(result_df)
