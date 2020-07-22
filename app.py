#!/usr/bin/python3

import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error

# Constants and mappings

DEFAULT_RANDOM_STATE = 88
BOOL_OPTIONS = [True, False]
ALGO_MAPPING = {'00': LinearRegression(), '01': KNeighborsRegressor(), '02': DecisionTreeRegressor(), '03': RandomForestRegressor(),
                '10': LogisticRegression(), '11': KNeighborsClassifier(), '12': DecisionTreeClassifier(), '13': RandomForestClassifier()}

LINREG_MAPPING = {'fit_intercept': BOOL_OPTIONS, 'normalize': BOOL_OPTIONS}
LOGREG_MAPPING = {'fit_intercept': BOOL_OPTIONS, 'penalty': ['l2', 'l1', 'elasticnet', 'none'], 'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga']}
KNN_MAPPING = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [2, 3, 4, 5, 6, 7, 8], 'weights': ['uniform', 'distance']}
DTREG_MAPPING = {'criterion': ["mse", "friedman_mse", "mae"], 'max_features': [None, "auto", "sqrt", "log2"]}
DTCLASS_MAPPING = {'criterion': ["gini", "entropy"], 'max_features': [None, "auto", "sqrt", "log2"]}
RFREG_MAPPING = {'n_estimators': [100, 200, 300], 'criterion': ['mse', 'mae'], 'max_features': [None, "auto", "sqrt", "log2"], 'bootstrap': BOOL_OPTIONS, 'oob_score': BOOL_OPTIONS}
RFCLASS_MAPPING = {'n_estimators': [100, 200, 300], 'criterion': ['gini', 'entropy'], 'max_features': [None, "auto", "sqrt", "log2"], 'bootstrap': BOOL_OPTIONS, 'oob_score': BOOL_OPTIONS}

ALGO_PARAMS_MAPPING = {'00': LINREG_MAPPING, '01': KNN_MAPPING, '02': DTREG_MAPPING, '03': RFREG_MAPPING,
                '10': LOGREG_MAPPING, '11': KNN_MAPPING, '12': DTCLASS_MAPPING, '13': RFCLASS_MAPPING}

st.title("Try Every Machine Learning Algorithms")
st.text("Often, as a Data Scientist we are very lazy in trying all the different ML algorithms\nfor a given dataset.")

data_file = st.file_uploader("Upload your dataset", type='csv')
code_parts = list()

if data_file is not None:
    data = pd.read_csv(data_file, sep=';')
    code_part_ld = "data = pd.read_csv(<filename>)"
    code_parts.append(code_part_ld)

    if st.checkbox("Show Raw Data", value=True):
        st.dataframe(data)

    if st.checkbox("Inspect Data"):
        st.text("Some of the descriptive statistics of the columns")
        st.dataframe(data.describe())

    st.subheader("Fitting the model")

    features = st.multiselect("Features", options=data.columns)
    labels = st.selectbox("Labels", options=data.columns, index=len(data.columns)-1)

    X = data[features]
    y = data[labels]

    code_part_f = "X = data["+str(features)+"]"
    code_part_l = "y = data[[\'"+str(labels)+"\']]\n"
    code_parts.extend([code_part_f, code_part_l])

    train_size = st.slider("Train Test Split", value=0.7, format="%f")
    code_part_tts = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size="+str(1-train_size)+")\n"
    code_parts.append(code_part_tts)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=DEFAULT_RANDOM_STATE)

    problem = st.selectbox("What type of Problem are you dealing with?", ['Regression', 'Classification'])

    common_algos = ["K Nearest Neighbours", "Descision Trees", "Random Forest"]
    if problem == "Regression":
        algos = ["Linear Regression"]+common_algos
        algo_name = st.selectbox(problem+" Algorithm", algos)

        model_code = '0'+str(algos.index(algo_name))
    elif problem == "Classification":
        algos = ["Logistic Regression"]+common_algos
        algo_name = st.selectbox(problem+" Algorithm", algos)

        model_code = '1'+str(algos.index(algo_name))

    st.sidebar.subheader("Hyperparametr  tuning")
    params = dict()
    for name, value in ALGO_PARAMS_MAPPING[model_code].items():
        params[name] = st.sidebar.selectbox(name, value)

    model = ALGO_MAPPING[model_code]
    model.set_params(**params)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    st.header("Evaluating the model's performance")
    if problem == "Regression":
        training_loss = mean_squared_error(y_pred_train, y_train)
        testing_loss = mean_squared_error(y_pred_test, y_test)

        code_part_tls = "training_loss = mean_squared_error(y_pred_train, y_train)\ntesting_loss = mean_squared_error(y_pred_test, y_test)\n"
        code_part_tlp = "printf(\"Training Loss is\", training_loss)\nprintf(\"Testing Loss is\", testing_loss)\n"
        code_parts.extend([code_part_tls, code_part_tlp])

        st.write("Training Loss is", training_loss)
        st.write("Testing Loss is", testing_loss)
    elif problem == "Classification":
        st.subheader("Accuracy Score")
        training_acc = accuracy_score(y_train, y_pred_train)
        testing_acc = accuracy_score(y_test, y_pred_test)

        code_part_tas = "training_acc = accuracy_score(y_train, y_pred_train)\ntesting_acc = accuracy_score(y_test, y_pred_test)\n"
        code_part_tap = "print(\"Training Accuracy is\", training_acc)\nprint(\"Testing Accuracy is\", testing_acc)\n"
        code_parts.extend([code_part_tas, code_part_tap])
        
        st.write("Training Accuracy is", training_acc)
        st.write("Testing Accuracy is", testing_acc)

        st.subheader("Confusion Matrix")
        con_mat = confusion_matrix(y_test, y_pred_test)
        st.table(con_mat)

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred_test)
        st.write(report)

        code_part_cm = "con_mat = confusion_matrix(y_test, y_pred_test)\nprint(con_mat)\n"
        code_part_cr = "report = classification_report(y_test, y_pred_test)\nprint(report)\n"
        code_parts.extend([code_part_cm, code_part_cr])










    if st.checkbox("Show the code"):
        implement_code = ""
        for code in code_parts:
            implement_code = implement_code + "\n" + code
        st.code(implement_code)