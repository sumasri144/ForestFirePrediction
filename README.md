<div id="top"></div>

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)



<!-- ABOUT THE PROJECT -->
## About The Project
* Using Data Science and Machine learning, we can build a model that takes in the detected fires dataset learns and detects future fires based on certain Weather report.
* Storing the Sourced dataset to MongoDB.
* Building a **Flask App** hosted on **Heroku**.
* **Sklearn** for pre-processing and Model Building
* Pandas, Numpy, Matplotlib for csv reading, Data Processing, Data Cleaning, Visualization etc.


<!-- GETTING STARTED -->
## Introduction
*  I used a dataset on **Algerian Forest Fires from UCI**. The dataset contains a culmination of forest fire observations and data in two regions of Algeria: the Bejaia region and the Sidi Bel-Abbes region. 
* The timeline of this dataset is from June 2012 to September 2012. In this project, we focused on whether certain weather features could predict forest fires in these regions using few Machine Learning algorithms. 

<!-- USAGE EXAMPLES -->
## Steps

* Installing Python, PyCharm, Monogodb, Git to Computer.
* Creating Flask app by importing `Flask` module.
* Download the source dataset from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++#).
* For Classification algorithm decided to predict the features `Classes` from the dataset which is Binary classification `(fire, not fire)`.
* For Regression Problem algorithm decided to predict the feature `FWI` (Fire weather Index) which is 90%+ correlated to Classes Feature.

### Loading CSV and Inserting to DB
* The Downloaded CSV file is loaded as pandas Dataframe using Pandas Library.
* Pandas Dataframe is converted to Dict .
* Mongodb Altas is used as DB here, with `pymongo library` mongodb is connected to python.
* Database and collections created via python and the list of dictionaries is uploaded using `collection.insert_many` method.

<p align="right">(<a href="#top">back to top</a>)</p> 

### EDA
* In this step, we will apply Exploratory Data Analysis (EDA) to extract insights from the data set to know which features have contributed more in predicting Forest fire by performing Data Analysis using Pandas and Data visualization using Matplotlib & Seaborn. 
* It is always a good practice to understand the data first and try to gather as many insights from it.

### Model Building 
* For Regression Problem algorithm decided to predict the feature `FWI` (Fire weather Index) which is 90%+ correlated to Classes Feature.
* Models used : **Linear regression, Lasso Regression, Ridge Regression, Random forest, Decision tree, K-Nearest Neighbour regressor, Support Vector Regressor.**
* For Classification algorithm decided to predict the features `Classes` from the dataset which is Binary classification `(fire, not fire)`.
* Models used : **Logistic Regression, Decision Tree, Random Forest, XGboost, K-Nearest Neighbour.**

### Model Selection
* HyperParameter Tuning Randomized Gridsearch CV is done for top 2 models for both Regression and Classification.
* For Classification `Stratified Kfold Cross-Validation metrics` is used based best Mean CV Accuracy Model is used for Model Deployment.
* For Regression `R2 score metrics` is used to select best model The R2 score is one of the performance evaluation measures for regression-based machine learning models.


<p align="right">(<a href="#top">back to top</a>)</p>

### **Technologies used**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)


### **Tools used**
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)


