# Diabetes Classification Project

## Introduction
This project is based on the paper _**'A data-driven approach to predicting diabetes and cardiovascular disease with machine learning'**_, written by BMC Medical Informatics and Decision Making (https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0918-5). The aim of the project was to replicate what the paper did with a focus on classifying and predicting **diabetes**. This includes the entire process of downloading raw data, exploring and understanding it before eventually using it to train models to predict whether a patient is diabetic, prediabetic, or not diabetic.

## Data
### Source: NHANES
The datasets used by the paper mentioned above, as well as in this project, were taken directly from the NHANES (National Health and Nutrition Examination Survey) under the Centers for Disease Control and Prevention (https://www.cdc.gov/) in the United States. This is an exhaustive study done over various year brackets and include patient data in the following categories; Demograpahic, Dietary, Examination, Laboratory, and Questionnaire. 

### Fetching and Storing NHANES data
The steps to obtain raw data were as follows:
1. From https://www.cdc.gov/nchs/nhanes/index.htm, click on _'Questionnaires, Datasets, and Related Documentation'_
2. From the side bar, start by clicking on 'NHANES 1999-2000'
3. For each "root category" (Demographic Data, Dietary Data, Examination Data, Laboratory Data, Questionnaire Data), download the Doc (data) file and DEMO (description) file.
4. Save all these in separate folders 
5. Repeat from Step 2 onwards for all other year brackets until 2017-2018

e.g. a sample file path inside the NHANES 1999-2000 folder would look like: 

_NHANES > NHANES 1999-2000 > Demographics data > Demographic Variables & Sample Weights > Data file > 'filename.xpt'_

## From Raw data to Pandas Dataframe
All the data files downloaded have the .xpt extension so in order to view these files, SAS Universal Viewer was used. 
The steps followed to create dataframes for each data file were as follows:
1. Located data files with multiple rows per patient and stored these file names in a list.
2. Went through the data files whose names weren't in the list above and merged all the data for one year bracket.
3. Went through the list of file names with multiple rows per patient and selected the files with data that should be added to the dataframe for each year bracket. 
4. The files chosen were those that contained data on prescription medications and dietary supplements.
5. For each year bracket (1999-2000, 2001-2002, ..., 2017-2018), two separate dataframes were created with data on patients' prescription medications and dietary supplements. 
6. These dataframes were then merged with the main dataframe for each year bracket, creating a "complete dataframe" for each year bracket.
7. A master dataframe was then created using the first occurrence of each patient in each complete dataframe for each year bracket. This only keeps patients whose diabetes outcome label can be classified using the same criteria as the paper. This criteria requires the patient to be _over the age of 19_ **and** a _male or a non-pregnant female_.
8. Used this master dataframe to create two dataframes to use in two approaches to this project; 1) Data-driven approach 2) Domain-driven approach

This is done by running the _**main.py**_ file one time and saving the dataframes created. (uses _**implementation_final.py**_ to run)

## Preprocessing Dataframe for Machine Learning using Transformation Pipelines
Create four machine learning transformation pipelines to test which performs better on dataset. A brief overview is shown below.
![github-pipelines-chart](https://user-images.githubusercontent.com/76870222/129478688-0c4e4f19-f7ca-4a06-a4e6-e6c3516cec6c.jpg)

## Data-driven Approach
For this first approach, the columns of the master dataframe were filtered even further using a maximum threshold of **0.5** for the percentage of NaN values present in each column. All columns with a percentage of NaN values greater than 0.5 were dropped from the master dataframe. The aim of this approach was to use as little domain knowledge as possible when it came to identifying the most relevant feature columns to keep and which ones to drop. Only columns that were redundant, repetitive, or unnecessary were removed. 

### Preparing Dataframe for Machine Learning
In this step, two experiments were done to model what was done in the BMC paper; we looked at two timeframes (1999-2014 and 2003-2014) and the inclusion and exclusion of lab data.
The experiments were as follows:

1. Clean dataframe, only keep relevant columns.
2. Sort columns based on whether they contain continuous, categorical, mixed, or object data.
3. Assign diabetes class labels (not diabetic - 0 | diabetic - 1 | prediabetic - 2)

- With Lab Data => 1999-2014 and 2003-2014
4. Restrict dataset to patients from 1999-2014 year brackets (then, patients from 2003-2014 are considered).
5. Drop columns not considered features to classify diabetes.
6. Separate predictors, X, and labels, y, which are used to train ML models.

- Without Lab Data => 1999-2014 and 2003-2014
4. Restrict dataset to patients from 1999-2014 year brackets (then, patients from 2003-2014 are considered).
5. Drop columns not considered features to classify diabetes as well as columns with lab data.
6. Separate predictors, X, and labels, y, which are used to train ML models.

### Machine Learning
Two machine learning classifiers were used in this project - **Random Forest Classifier** (RFC) and **XGBoost** (Extreme Gradient Boosting) **Classifier** (XGB). For each classifer, the following was done:

1. Use each transformation pipeline above to fit and transform X. 
2. Fit a baseline classifier to the tranformed X and y.
3. Evaluate using 5-fold cross validation.
4. Calculate and compare mean cross validation scores and ROC AUC scores.
5. The pipeline which results in the highest ROC AUC score is stored as the best transformation pipeline for that classfier and will be used in grid search.
6. Calculate and compare mean cross validation scores and ROC AUC score once again.

#### Best model
The best model is selected as the classifier that resulted in the highest ROC AUC score. In this data-driven approach, it is the XGB Classifier after conducting grid search with a parameter grid. Then, using the XGBoost feature importance method, the top 20 features used in classification were identified.

## Domain-driven Approach
In this approach, the master dataframe was filtered using a maximum threshold of **0.55** for the percentage of NaN values present in each column. This led to the inclusion of a number of more "relevant" columns as well as a lot of extra columns deemed "irrelevant" being dropped. Feature engineering was also used in this approach to combine multiple feature columns under the same group to create more conclusive dummy feature columns. 

### Preparing Dataframe for Machine Learning
1. Clean the dataframe. Only columns that were deemed relevant to the classification of diabetes were kept. 
2. Perform feature engineering to reduce the number of feature columns in the dataframe by grouping columns under the same category into one combined dummy feature.
3. Sort columns based on whether they contain continuous, categorical, mixed, or object data.
4. Assign diabetes class labels (not diabetic - 0 | diabetic - 1 | prediabetic - 2)

- With Lab Data => 1999-2014 and 2003-2014
5. Restrict dataset to patients from 1999-2014 year brackets (then, patients from 2003-2014 are considered).
6. Drop columns not considered features to classify diabetes.
7. Separate predictors, X, and labels, y, which are used to train ML models.

- Without Lab Data => 1999-2014 and 2003-2014
5. Restrict dataset to patients from 1999-2014 year brackets (then, patients from 2003-2014 are considered).
6. Drop columns not considered features to classify diabetes as well as columns with lab data.
7. Separate predictors, X, and labels, y, which are used to train ML models.

### Machine Learning
The same steps were followed as in the Data-driven approach.

#### Best model
The best model is selected as the classifier that resulted in the highest ROC AUC score. In this domain-driven approach, it is the XGB Classifier after conducting grid search with a parameter grid. Then, using the XGBoost feature importance method, the top 20 features used in classification were identified.

### Comparing Results to BMC Paper's Results
The table below compares the ROC AUC scores of the best performing models in the BMC paper with the best performing ones in the two approaches in this project. 

## Diagram of Flow of Python files
![G42-Diabetes-project-github-project-flow-chart](https://user-images.githubusercontent.com/76870222/129860407-7953bd0e-2927-417c-ae0b-697a605d8a06.jpg)

1. Run the _**main.py**_ file first. This file will import functions and variables from _**implementation_final.py**_ automatically.
2. In the **With Lab Data** experiment:
- For the **Data-driven approach**, run the _**_03_with_lab_data_driven_best_model.py**_ file. This file will import functions and variables from the _**_01_with_lab_data_driven_prep_for_ML.py**_ and _**_02_with_lab_data_driven_test_pipelines.py**_ files automatically.
- For  the **Domain-driven approach**, run the _**_06_with_lab_domain_driven_best_model.py**_ file. This file will import functions and variables from the _**_04_with_lab_domain_driven_prep_for_ML.py**_ and _**_05_with_lab_domain_driven_test_pipelines.py**_ files automatically.
3. In the **Without Lab Data** experiment:
- For the **Data-driven approach**, run the _**_03_without_lab_data_driven_best_model.py**_ file. This file will import functions and variables from the _**_01_without_lab_data_driven_prep_for_ML.py**_ and _**_02_without_lab_data_driven_test_pipelines.py**_ files automatically.
- For  the **Domain-driven approach**, run the _**_06_without_lab_domain_driven_best_model.py**_ file. This file will import functions and variables from the _**_04_without_lab_domain_driven_prep_for_ML.py**_ and _**_05_without_lab_domain_driven_test_pipelines.py**_ files automatically.


## Backlog
- Play around with values for the maximum percentage of NaN value threshold when filtering columns in the master dataframe and investigate whether adding more or less features would improve model performance.
- Play around with various other transformation pipelines - using different methods of imputation (MICE - correlated categorical features). Imputation is most probably the biggest reason for the difference in performance between the models in this project and the BMC paper. The paper doesn't mention the methods used to impute missing values.
- Perform multiple Grid Searches with RFC and XGB and test different hyperparameter values.
- Change the goal of the project to classifying cardiovascular disease.
