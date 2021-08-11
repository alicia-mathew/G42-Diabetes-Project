# Markdown Demo

## Introduction
This project is based on the paper _**'A data-driven approach to predicting diabetes and cardiovascular disease with machine learning'**_, written by BMC Medical Informatics and Decision Making (https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-019-0918-5). The aim of the project was to replicate what the paper did with a focus on classifying and predicting **diabetes**. This includes the entire process of downloading raw data, exploring and understanding it before eventually using it to train models to predict whether a patient is diabetic, prediabetic, or not diabetic.

## Data
### Source: NHANES
The datasets used by the paper mentioned above, as well as in this project, were taken directly from the NHANES (National Health and Nutrition Examination Survey) under the Centers for Disease Control and Prevention (https://www.cdc.gov/) in the United States. This is an exhaustive study done over various year brackets and include patient data in the following categories; Demograpahic, Dietary, Examination, Laboratory, and Questionnaire. 

### Fetching NHANES data
### Storing NHANES data
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


## Preprocessing Dataframe for Machine Learning

## Machine Learning

## Backlog

## Access to Models
## Diagram of Models

