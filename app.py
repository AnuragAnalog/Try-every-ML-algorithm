#!/usr/bin/python3

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

DEFAULT_RANDOM_STATE = 88
BOOL_OPTIONS = [True, False]
ALGO_MAPPING = {}

st.title("Try Every Machine Learning Algorithms")
st.text("Often, as a Data Scientist we are very lazy in trying all the different ML algorithms\nfor a given dataset.")

data_file = st.file_uploader("Upload your dataset", type='csv')

if data_file is not None:
    data = pd.read_csv(data_file)
    code_part_ld = "data = pd.read_csv(<filename>)"

    if st.checkbox("Show Raw Data", value=True):
        st.dataframe(data)

    if st.checkbox("Inspect Data"):
        st.text("Some of the descriptive statistics of the columns")
        st.dataframe(data.describe())

    features = st.multiselect("Features", options=data.columns)
    labels = st.selectbox("Labels", options=data.columns, index=len(data.columns)-1)

    X = data[features]
    code_part_f = "X = data["+str(features)+"]"

    y = data[labels]
    code_part_l = "y = data[[\'"+str(labels)+"\']]"

    train_size = st.slider("Train Test Split", value=0.7, format="%f")
    st.write(train_size)
    code_part_tts = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size="+str(1-train_size)+")"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=DEFAULT_RANDOM_STATE)

    problem = st.selectbox("What type of Problem are you dealing with?", ['Regression', 'Classification'])

    common_algos = ["K Nearest Neighbours", "Descision Trees", "Random Forest"]
    if problem == "Regression":
        algos = ["Linear Regression"]
        algo_name = st.selectbox(problem+" Algorithm", algos+common_algos)
    elif problem == "Classification":
        algos = ["Logistic Regression"]
        algo_name = st.selectbox(problem+" Algorithm", algos+common_algos)

    st.sidebar.subheader("Hyperparametr  tuning")
    param1 = st.sidebar.selectbox('fit_intercept', BOOL_OPTIONS, index=0)
    param2 = st.sidebar.selectbox('normalize', BOOL_OPTIONS, index=1)

    model = LinearRegression(fit_intercept=param1, normalize=param2)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    st.write(y_pred)










    if st.checkbox("Show the code"):
        code_part = code_part_ld + "\n\n" + code_part_f + "\n" + code_part_l + "\n\n" + code_part_tts
        st.code(code_part)