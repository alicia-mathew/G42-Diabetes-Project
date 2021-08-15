# Markdown Demo

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
8. Used this master dataframe to create two dataframes to use in two approaches to this project.

This is done by running the **main.py** file one time and saving the dataframes created.

### Data-driven Approach
For this first approach, the columns of the master dataframe were filtered even further using a maximum threshold of **0.5** for the percentage of NaN values present in each column. All columns with a percentage of NaN values greater than 0.5 were dropped from the master dataframe. The aim of this approach was to use as little domain knowledge as possible when it came to identifying the most relevant feature columns to keep and which ones to drop. Only columns that were redundant, repetitive, or unnecessary were removed. 

### Domain-driven Approach
In this approach, the master dataframe was filtered using a maximum threshold of **0.55** for the percentage of NaN values present in each column. This led to the inclusion of a number of more "relevant" columns as well as a lot of extra columns deemed "irrelevant" being dropped. Feature engineering was also used in this approach to combine multiple feature columns under the same group to create more conclusive dummy feature columns. 

## Preprocessing Dataframe for Machine Learning

## Machine Learning

## Backlog

## Access to Models
## Diagram of Models

