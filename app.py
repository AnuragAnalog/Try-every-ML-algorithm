#!/usr/bin/python3

import pandas as pd
import streamlit as st

from PIL import Image
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

# Suppress streamlit warning's
st.set_option('deprecation.showfileUploaderEncoding', False)

# Constants and mappings

DEFAULT_RANDOM_STATE = 88
BOOL_OPTIONS = [True, False]
ALGO_MAPPING = {'00': LinearRegression(n_jobs=-1), '01': KNeighborsRegressor(n_jobs=-1), '02': DecisionTreeRegressor(), '03': RandomForestRegressor(n_jobs=-1), '04': AdaBoostRegressor(), '05': GradientBoostingRegressor(),
                '10': LogisticRegression(n_jobs=-1), '11': KNeighborsClassifier(n_jobs=-1), '12': DecisionTreeClassifier(), '13': RandomForestClassifier(n_jobs=-1), '14': AdaBoostClassifier(), '15': GradientBoostingClassifier()}

LINREG_MAPPING = {'fit_intercept': BOOL_OPTIONS, 'normalize': BOOL_OPTIONS}
LOGREG_MAPPING = {'C': [None, 1.0, 2.0, 0.01], 'fit_intercept': BOOL_OPTIONS, 'penalty': ['l2', 'l1', 'elasticnet', 'none'], 'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'], 'max_iter': [None, 100, 500, 10], 'multi_class': ['auto', 'ovr', 'multinomial'], 'warm_start': BOOL_OPTIONS}
KNN_MAPPING = { 'n_neighbors': [None, 2, 15], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [None, 30, 100, 10], 'p': [None, 2, 20], 'weights': ['uniform', 'distance']}
DTREG_MAPPING = {'criterion': ["mse", "friedman_mse", "mae"], 'splitter': ["best", "random"], 'max_depth': [None, 3, 20], 'max_features': [None, "auto", "sqrt", "log2"], 'min_impurity_decrease': [None, 0.0, 1.0, 0.05], 'min_samples_split': [None, 0.1, 1.0, 0.05], 'min_samples_leaf': [None, 0.1, 1.0, 0.05]}
DTCLASS_MAPPING = {'criterion': ["gini", "entropy"], 'splitter': ["best", "random"], 'max_depth': [None, 3, 20], 'max_features': [None, "auto", "sqrt", "log2"], 'min_impurity_decrease': [None, 0.0, 1.0, 0.05], 'min_samples_split': [None, 0.1, 1.0, 0.05], 'min_samples_leaf': [None, 0.1, 1.0, 0.05]}
RFREG_MAPPING = {'n_estimators': [None, 100, 1000, 50], 'criterion': ['mse', 'mae'], 'max_depth': [None, 3, 20], 'max_features': [None, "auto", "sqrt", "log2"], 'bootstrap': BOOL_OPTIONS, 'oob_score': BOOL_OPTIONS, 'min_impurity_decrease': [None, 0.0, 1.0, 0.05], 'min_samples_split': [None, 0.1, 1.0, 0.05], 'min_samples_leaf': [None, 0.1, 1.0, 0.05]}
RFCLASS_MAPPING = {'n_estimators': [None, 100, 1000, 50], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 3, 20], 'max_features': [None, "auto", "sqrt", "log2"], 'bootstrap': BOOL_OPTIONS, 'oob_score': BOOL_OPTIONS, 'min_impurity_decrease': [None, 0.0, 1.0, 0.05], 'min_samples_split': [None, 0.1, 1.0, 0.05], 'min_samples_leaf': [None, 0.1, 1.0, 0.05]}
ADAREG_MAPPING = {'n_estimators': [None, 50, 500, 50], 'learning_rate': [None, 1.0, 2.0, 0.05], 'loss': ['linear', 'square', 'exponential']}
ADACLASS_MAPPING = {'n_estimators': [None, 50, 500, 50], 'learning_rate': [None, 1.0, 2.0, 0.05], 'algorithm': ['SAMME', 'SAMME.R']}
GBREG_MAPPING = {'loss': ['ls', 'lad', 'huber', 'quantile'], 'learning_rate': [None, 0.1, 1.0, 0.05], 'n_estimators': [None, 50, 500, 50], 'criterion': ["friedman_mse", "mse", "mae"], 'alpha': [None, 0.9, 2.0, 0.05], 'validation_fraction': [None, 0.1, 0.8, 0.1], 'n_iter_no_change': [None, 50, 100, 5], 'max_features': [None, "auto", "sqrt", "log2"], 'min_impurity_decrease': [None, 0.0, 1.0, 0.05], 'min_samples_split': [None, 0.1, 1.0, 0.05], 'min_samples_leaf': [None, 0.1, 1.0, 0.05]}
GBCLASS_MAPPING = {'loss': ['deviance', 'exponential'], 'learning_rate': [None, 0.1, 1.0, 0.05], 'n_estimators': [None, 50, 500, 50], 'criterion': ["friedman_mse", "mse", "mae"], 'validation_fraction': [None, 0.1, 0.8, 0.1], 'n_iter_no_change': [None, 50, 100, 5], 'max_features': [None, "auto", "sqrt", "log2"], 'min_impurity_decrease': [None, 0.0, 1.0, 0.05], 'min_samples_split': [None, 0.1, 1.0, 0.05], 'min_samples_leaf': [None, 0.1, 1.0, 0.05]}

ALGO_PARAMS_MAPPING = {'00': LINREG_MAPPING, '01': KNN_MAPPING, '02': DTREG_MAPPING, '03': RFREG_MAPPING, '04': ADAREG_MAPPING, '05': GBREG_MAPPING,
                '10': LOGREG_MAPPING, '11': KNN_MAPPING, '12': DTCLASS_MAPPING, '13': RFCLASS_MAPPING, '14': ADACLASS_MAPPING, '15': GBCLASS_MAPPING}

LOSS_MAPPING = {'mse': mean_squared_error, 'mae': mean_absolute_error, 'msle': mean_squared_log_error}

st.title("Try Every Machine Learning Algorithms")
st.text("Often, as a Data Scientist we are very lazy in trying all the different ML algorithms\nfor a given dataset.")

data_file = st.file_uploader("Upload your dataset", type='csv')
code_parts = list()

if data_file is not None:
    data = pd.read_csv(data_file)
    code_part_ld = "data = pd.read_csv(<filename>)"
    code_parts.append(code_part_ld)

    if st.checkbox("Show Raw Data", value=True):
        st.dataframe(data)

    if st.checkbox("Inspect Data"):
        st.text("Some of the descriptive statistics of the columns")
        st.dataframe(data.describe())

    if data.isna().any().any():
        st.warning("Make sure that the dataset is NA free")
    else:
        st.success("You are good to go, the dataset is NA free")

    st.header("Preprocessing")
    encode_cols = st.multiselect("Features", options=data.columns, key='preprocessing')

    if len(encode_cols) != 0:
        st.subheader("Features which will be encoded are")
        st.write(encode_cols)

        data = encode_features(data, encode_cols)

    st.header("Fitting the model")

    features = st.multiselect("Features", options=data.columns, key='fitting')
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

    common_algos = ["K Nearest Neighbours", "Descision Trees", "Random Forest", "Ada Boost", "Gradient Boosting"]
    if problem == "Regression":
        algos = ["Linear Regression"]+common_algos
        algo_name = st.selectbox(problem+" Algorithm", algos)

        model_code = '0'+str(algos.index(algo_name))
    elif problem == "Classification":
        algos = ["Logistic Regression"]+common_algos
        algo_name = st.selectbox(problem+" Algorithm", algos)

        model_code = '1'+str(algos.index(algo_name))

    st.sidebar.title(problem)
    st.sidebar.header(algo_name)
    st.sidebar.subheader("Hyperparameter  Tuning")

    params = dict()
    for name, value in ALGO_PARAMS_MAPPING[model_code].items():
        if value[0] is None:
            if isinstance(value[1], Union[int, float].__args__) and len(value) == 4:
                params[name] = st.sidebar.number_input(name, min_value=value[1], max_value=value[2], step=value[3])
            elif isinstance(value[1], Union[int, float].__args__) and len(value) == 3:
                params[name] = st.sidebar.slider(name, min_value=value[1], max_value=value[2])
            else:
                params[name] = st.sidebar.selectbox(name, value)
        else:
            params[name] = st.sidebar.selectbox(name, value)

    model = ALGO_MAPPING[model_code]
    model.set_params(**params)
    model.fit(X_train, y_train)

    code_part_init = "model = "+str(model)+"\n"
    code_part_fit = "model.fit(X_train, y_train)\n"

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    code_part_predicts = "y_pred_train = model.predict(X_train)\ny_pred_test = model.predict(X_test)\n"
    code_parts.extend([code_part_init, code_part_fit, code_part_predicts])

    st.header("Evaluating the model's performance")
    if problem == "Regression":
        loss = st.selectbox("Select the type of loss", list(LOSS_MAPPING.keys()))

        training_loss = LOSS_MAPPING[loss](y_pred_train, y_train)
        testing_loss = LOSS_MAPPING[loss](y_pred_test, y_test)

        loss_str = str(LOSS_MAPPING[loss]).split()[1]
        code_part_tls = "training_loss = "+loss_str+"(y_pred_train, y_train)\ntesting_loss = "+loss_str+"(y_pred_test, y_test)\n"
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
        st.text("Code to implement the above algorithm")
        implement_code = ""
        for code in code_parts:
            implement_code = implement_code + "\n" + code
        st.code(implement_code)