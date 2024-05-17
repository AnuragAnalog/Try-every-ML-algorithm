#!/usr/bin/python3

import pandas as pd
import streamlit as st

from typing import Union

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier

# Required functions
def encode_features(data, cols, drop_original=True):
    if len(cols) == 1:
        data = pd.concat([data, pd.get_dummies(data, prefix=cols[0])], axis=1)
    else:
        data = pd.concat([data, pd.get_dummies(data)], axis=1)

    if drop_original:
        data.drop(labels=cols, axis=1, inplace=True)

    return data

# Setting page items
# favicon = Image.open('./favicon.png')

# st.beta_set_favicon(favicon, format='PNG')
st.set_page_config(page_title='Try Every Machine Learning Algorithms', page_icon=':robot:', layout='wide')

# Suppress streamlit warning's
st.set_option('deprecation.showfileUploaderEncoding', False)

# Constants and mappings
DEFAULT_RANDOM_STATE = 88

st.title("Try Every Machine Learning Algorithms")
st.text("Often, as a Data Scientist we are very lazy in trying all the different ML algorithms\nfor a given dataset.")

data_file = st.file_uploader("Upload your dataset", type='csv')

if data_file is not None:
    data = pd.read_csv(data_file)

    if st.checkbox("Show Raw Data", value=True):
        st.dataframe(data)

    if st.checkbox("Inspect Data"):
        st.text("Some Info about the dataset")
        st.dataframe(data.info())
        st.text("Some of the descriptive statistics of the columns")
        st.dataframe(data.describe())

    if data.isna().any().any():
        st.warning("Make sure that the dataset contains no missing values")
    else:
        st.success("You are good to go, the dataset doesn't have any missing values")

    st.header("Preprocessing")
    encode_cols = st.multiselect("Features", options=data.columns, key='preprocessing')

    if len(encode_cols) != 0:
        st.subheader("Features which will be encoded are")
        st.write(encode_cols)

        data = encode_features(data, encode_cols)