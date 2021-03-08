import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import shap
import xgboost
import matplotlib.pyplot as plt
import time


def run_status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Percent Complete {i+1}')
		bar.progress(i + 1)
		time.sleep(0.1)
		st.empty()


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('SteamFlow Prediction APP for a Papermil ')
from PIL import Image
image = Image.open('papermil1000.jpg')
st.image(image, caption='Paper mill- Paper production (Own image)', width=400)

# EDA
st.subheader('Exploratory Data Analysis')
my_dataset = "papermil_data_new.csv"

# To Improve speed and cache data
@st.cache(persist=True)
def explore_data(dataset):
	df = pd.read_csv(os.path.join(dataset))
	return df 

def user_input_features():
   
        a = st.sidebar.slider('a', 131.0,223.0,175.0,0.2)
        b = st.sidebar.slider('b', 2148.0,3700.0,2888.0,0.5)
        c = st.sidebar.slider('c', 26.0,531.0,330.0,0.2)    
        d = st.sidebar.slider('d', 26.0,531.0,330.0,0.2)                
        e = st.sidebar.slider('e', 26.0,531.0,330.0,0.2)                      
        f = st.sidebar.slider('f', 143.0,295.0,237.0,0.2)
        g = st.sidebar.slider('g', 143.0,287.0,237.0,0.2)       
        h = st.sidebar.slider('h', 143.0,290.0,237.0,0.2)
        i = st.sidebar.slider('i', 2.0,100.0,75.0,0.2)
        j = st.sidebar.slider('j', 0.0,14.0,10.0,0.2)
        k = st.sidebar.slider('k', 115.0,205.0,160.0,0.2)
   
       
       
        
        data = {'a': a,             
                'b': b,           
                'c': c,
                'd': d,
                'e': e,             
                'f': f,
                'g': g,
                'h': h,            
                'i': i,
                'j': j,
                'k': k           
               }
        features = pd.DataFrame(data, index=[0])
        return features
   


# Show Entire Dataframe
if st.checkbox("Show Dataset used in the Model Building"):
	data = explore_data(my_dataset)
	st.dataframe(data)

# Show Description
if st.checkbox("Show All Column Names"):
	data = explore_data(my_dataset)
	st.text("Columns:")
	st.write(data.columns)
    
# Dimensions
data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
if data_dim == 'Rows':
	data = explore_data(my_dataset)
	st.text("Showing Length of Rows")
	st.write(len(data))
if data_dim == 'Columns':
	data = explore_data(my_dataset)
	st.text("Showing Length of Columns")
	st.write(data.shape[1])


if st.checkbox("Show Summary of Dataset"):
	data = explore_data(my_dataset)
	st.write(data.describe())
    

showlinechart= st.checkbox('Show line chart for SteamFlow')   
    
if showlinechart:
    st.line_chart(data['steamflow'])



showdescription = st.checkbox('Show Project Description')

if showdescription:
    st.write("""
# Description

This app predicts the **SteamFlow** of a PaperRoll in PaperMill!

During the production of a paper roll in the paper mill, it takes around **17-20** tons of water/hour(Depends on the size of the roll).
There are 403 parameters nobs are all around the paper machine, which measures different parameters during the production. Among them, 11 are
more important that has good effect on the water uses(SteamFlow) parameter.

With this app production manager can check how much water will be used to produce a paper roll and set the parameters to those 12 nobs.

#  Tools used:

We have used Linear regression, XGBoost, Random Forest, Decision tree 
  

Data we had collected from a paper mill open data. The data consists of 3941 records with 11(independent) 1(dependent) features. 

Used Streamlit Library.

""")

st.sidebar.header('User Input Features- 11 parameters: Change the parameters and see the result in the right side')


input_df = user_input_features()
#as Random forest gives us the minimum error(RMSE), we will use that to 
#find the prediction of Steamflow


# # Reads in saved Linear Model
# load_linear_model = pickle.load(open('papermil_lreg.pkl', 'rb'))


# # Apply Linear model to make predictions
# prediction_linear = load_linear_model.predict(input_df)

# Reads in saved XGB  model
load_xgb_model = pickle.load(open('papermil_xgb.pkl', 'rb'))

# Apply XGB model to make predictions
prediction_xgb = load_xgb_model.predict(input_df)

# # Reads in saved Random Forest  model
# load_RF_model = pickle.load(open('papermil_rf.pkl', 'rb'))


# # Apply RF model to make predictions
# prediction_RF = load_RF_model.predict(input_df)

# # Reads in saved Decsion Tree  model
# load_DT_model = pickle.load(open('papermil_dt.pkl', 'rb'))


# # Apply RF model to make predictions
# prediction_DT = load_DT_model.predict(input_df)

## Result displaying in Table
resultframe = { #'Linear_Regression': [prediction_linear],
                'XGBoost':  [prediction_xgb]
                #'Random_Forest': [prediction_RF],
                #'Decision_tree': [prediction_DT]
        }

df_res = pd.DataFrame (resultframe, columns = ['XGBoost'])

st.write("""
         # Below is the Result :(Tons/hour)""")
st.write(df_res)


#Model Explainability--

st.write("""
         # Model Explainability- Random Forest """)

st.subheader("""
# Why it is important?""")

st.write("""
Machine learning can't be a black box today, as GDPR implemented, model should have the power to explain.
By which companies can built trust among user and customer using a machine learning model.

""")         
         
# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap

explainer = shap.TreeExplainer(load_xgb_model)
shap_values = explainer.shap_values(input_df)



st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
st.text('Explaining the model predictions using SHAP values')
st.write("SHAP in Github [link](https://github.com/slundberg/shap)")
run_status()
shap.summary_plot(shap_values, input_df)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, input_df, plot_type="bar")
st.pyplot(bbox_inches='tight')



st.subheader("Model-Built with Random Forest, Deployed with Streamlit")
st.text("by: Ricky D'Cruze")
st.write("Source code, Data, pickle file,notebook in Github [link](https://github.com/rickystanley76/BTH-ML-with-streaming-data)")   


