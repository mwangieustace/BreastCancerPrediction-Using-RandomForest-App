import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sklearn
from streamlit_option_menu import option_menu
import joblib


def main():
    st.title('Breast Cancer Prediction Model Using Random Forest')
    filename = 'rfr_model.pkl'
    loaded_model = joblib.load(filename)
        #Caching the model for faster loading
         # @st.cache
        
    radius_mean = st.number_input('Insert radius_mean')
    texture_mean = st.number_input('Insert texture_mean')
    perimeter_mean = st.number_input('Insert perimeter_mean')
    area_mean = st.number_input('Insert area_mean')
    smoothness_mean = st.number_input('Insert smoothness_mean')
    compactness_mean = st.number_input('Insert compactness_mean')
    concavity_mean = st.number_input('Insert concavity_mean')
    concave_points_mean = st.number_input('Insert concave points_mean')
    symmetry_mean = st.number_input('Insert symmetry_mean')
    radius_se = st.number_input('Insert radius_se')
    perimeter_se = st.number_input('Insert perimeter_se')
    area_se = st.number_input('Insert area_se')
    compactness_se = st.number_input('Insert compactness_se')
    concavity_se = st.number_input('Insert concavity_se')
    concave_points_se = st.number_input('Insert concave points_se')
    radius_worst = st.number_input('Insert radius_worst')
    texture_worst = st.number_input('Insert texture_worst')
    perimeter_worst = st.number_input('Insert perimeter_worst')
    area_worst = st.number_input('Insert area_worst')
    smoothness_worst = st.number_input('Insert smoothness_worst')
    compactness_worst = st.number_input('Insert compactness_worst')
    concavity_worst = st.number_input('Insert concavity_worst')
    symmetry_worst = st.number_input('Insert symmetry_worst')
    concave_points_worst = st.number_input('Insert concave points_worst')
    fractal_dimension_worst = st.number_input('Insert fractal_dimension_worst')
    


    input_dict = {'radius_mean':radius_mean, 'texture_mean':texture_mean, 'perimeter_mean':perimeter_mean,
       'area_mean':area_mean, 'smoothness_mean':smoothness_mean, 'compactness_mean':compactness_mean, 'concavity_mean':concavity_mean,
       'concave points_mean':concave_points_mean, 'symmetry_mean':symmetry_mean, 'radius_se':radius_se, 'perimeter_se':perimeter_se,
       'area_se':area_se, 'compactness_se':compactness_se, 'concavity_se':concavity_se, 'concave points_se':concave_points_se,
       'radius_worst':radius_worst, 'texture_worst':texture_worst, 'perimeter_worst':perimeter_worst, 'area_worst':area_worst,
       'smoothness_worst':smoothness_worst, 'compactness_worst':compactness_worst, 'concavity_worst':concavity_worst,
       'concave points_worst':concave_points_worst, 'symmetry_worst':symmetry_worst, 'fractal_dimension_worst':fractal_dimension_worst}
            
    input_df = pd.DataFrame(input_dict, index=[0])    
            
            ## predict button
    button = st.button('Predict')
    #Benign tumors are noncancerous. Malignant tumors are cancerous.
    # benign==0, malig==1 
    if button:

            risk = loaded_model.predict(input_df)
            if risk == 0:
                st.success("tumor is benign")
            else:
                st.error("tumor is malignant")
            
            precision, recall, f1, acc = st.columns(4)
            st.markdown("""
            <style>
            div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 5% 5% 5% 10%;
            border-radius: 5px;
            color: rgb(30, 103, 119);
            overflow-wrap: break-word;
            }

            /* breakline for metric text         */
            div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
            overflow-wrap: break-word;
            white-space: break-spaces;
            color: green;
            font-size: 20px;
            }
            </style>
            """
            , unsafe_allow_html=True)

            with precision:
                st.metric(label="Precision Score", value="94%")
            with recall:
                st.metric(label="Recall Score", value="94%")
            with f1:
                st.metric(label="F1 Score", value="94%")
            with acc:
                st.metric(label="Accuracy Score", value="94%")


if __name__ == '__main__':
    main()