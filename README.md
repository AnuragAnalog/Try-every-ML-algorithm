# Try every ML algorithm

We as a Data Scientist are very lazy in trying out every different algorithm on a given dataset.

This web interface provides you a convinent way of switching between algorithms are seeing there results. With this now, you can apply machine learning models without writing a Single Piece of code.

The web application is written in streamlit.

Link to the web app is [here](https://share.streamlit.io/anuraganalog/try-every-ml-algorithm/app.py)

## How it works

* Open the App
* Upload the dataset*
* Inspect your data, if you wish
* Select the features
* Select the label
* Make a Train Test split
* Select an appropriate algorithm
* Start tweaking the hyperparameters

> This app expects a preprocessed dataset with all the NaN, Null values handled properly, One Hot encoded, and scaled

## Demo
![demo](./demo.gif)

## Required Modules

* Pandas
* Streamlit
* Scikit-learn

## Getting a copy of this repo
Clone the repository before running any commands
```python3
$ git clone https://github.com/AnuragAnalog/Try-every-ML-algorithm.git
$ cd Try-every-ML-algorithm
```

## Installation
Run the below command to install all the dependencies in your local machine to run the py script.

```python3
$ sudo pip3 install -r requirements.txt
```

## Running the app
```python3
$ streamlit run app.py
```

## Algorithms

* Regression
    * Linear Regression
    * K Nearest Neighbours
    * Decision Trees
    * Random Forest
    * Ada Boost
    * Gradient Boosting

* Classification
    * Logistic Regression
    * K Nearest Neighbours
    * Decision Trees
    * Random Forest
    * Ada Boost
    * Gradient Boosting

*Want to contribute? then fork, develop, and create a pull-request*

## Future Work

* [x] Add an option to show the code which implements the above selected algorithm with the corresponding hyperparameters.
* [x] Added code for One Hotencoding.
* [ ] Add more algorithms.
* [ ] Add some functionality for preprocessing data too.
