#%%
import pandas as pd
import numpy as np
import os
from pathlib import Path

root_loc = os.path.abspath("..") # '/home/alicia_mathew/project'

def locate_xpt_files(loc):
    # Takes a folder and returns .xpt files located anywhere
    # within this folder and its subfolders
    assert os.path.isdir(loc), "This function needs a valid direct, you \
        specified {}, check".format(loc)   
    flist = list(Path(loc).rglob("*.[xX][pP][tT]"))
    return flist

from collections import Counter

def find_common_columns(years, root_cat):
    # Takes a list of years and a string representing a root category 
    # (Demographics, Dietary, Examination, Laboratory, or Questionnaire) 
    # and outputs a (potentially) condensed list of columns from that 
    # root category 
    assert(len(years) != 0)
    #root_loc = '/mnt/c/Users/alicia.mathew/Desktop/Project/NHANES data'

    # Dict of years that stores unique list of columns present in those years
    dict_of_columns = {}

    for year in years:
        data_dir = os.path.join(root_loc, 'diabetes_project/datasets', 'NHANES '+year, root_cat)
        
        # Get a list of XPT files from data_dir
        xpt_list = locate_xpt_files(data_dir)

        # Create empty list for all the columns across each file
        all_file_columns = []

        for xpt_loc in xpt_list:
            temp_df = pd.read_sas(xpt_loc)
            if 'SEQN' in temp_df.columns:
                columns = list(temp_df.columns)
                all_file_columns.extend(columns)
        
        dict_of_columns[year] = list(set(all_file_columns)) # unique list
        #print(dict_of_columns[year])
        print('Finished {}'.format(year)) 

    all_columns = [] 
    for year in dict_of_columns:
        all_columns.extend(dict_of_columns[year]) # fix this
    
    # Max number of occurrences of column name if it appeared each year
    max_occurrence = len(dict_of_columns) 

    column_count = dict(Counter(all_columns))
    #print(column_count)
    common_columns = [column for column in column_count if \
        (column_count[column] == max_occurrence)] # fix this?

    return common_columns

def intersection(list1, list2):
    # Returns the intersection of two lists
    list3 = [elem for elem in list1 if elem in list2]
    return list3

def dict_SEQN(years, root_cat):
    # Returns a dict of SEQN and stores a list of years that SEQN appears in 
    #root_loc = '/mnt/c/Users/alicia.mathew/Desktop/Project/NHANES data'
    dict_of_SEQN_list = {}

    for year in years:
        data_dir = os.path.join(root_loc, 'diabetes_project/datasets', 'NHANES '+year, root_cat)

        # Get a list of XPT files from data_dir
        xpt_list = locate_xpt_files(data_dir)

        # All unique SEQN in given year
        SEQN_in_year = []

        for xpt_loc in xpt_list:
            temp_df = pd.read_sas(xpt_loc)
            if 'SEQN' in temp_df.columns:
                SEQN_in_file = list(temp_df['SEQN'])
                for seqn in SEQN_in_file:
                    if seqn not in SEQN_in_year:
                        SEQN_in_year.append(seqn)
        
        print('Finished {}'.format(year))
        dict_of_SEQN_list[year] = SEQN_in_year

    print('Finished looping through years.')
    print('There is a dict of lists of SEQNS present in each year.')

    dict_of_SEQN = {}

    for year in dict_of_SEQN_list:
        SEQN_list = dict_of_SEQN_list[year]
        for seqn in SEQN_list:
            if seqn not in dict_of_SEQN:
                dict_of_SEQN[seqn] = [year]
            else:
                dict_of_SEQN[seqn].append(year)

    print('There are a total of {} SEQN present over the years of {}.'.format(\
        len(dict_of_SEQN), root_cat))
    return dict_of_SEQN

#%%
def combine_same_columns(df):
    # Takes a df, with multiple columns that hold the same values (same column name)
    # Returns a new df with a column combining all the same columns' values
    # (gets rid of repearing column names)

    # List of columns that repeat (but hold the same values but have diff SEQN in their files)
    columns = list(df.columns)
    repeating_cols = []
    for column in columns:
        if '_' in column:
            repeating_col = column.split('_')[0]
            if repeating_col not in repeating_cols:
                repeating_cols.append(repeating_col)
    
    # keep dropping column from df after we store its values in a dict
    for a_repeating_col in repeating_cols:
        dict_seqn_repeating_col_vals = {}
        seq_values = list(df.iloc[:,0])
        for seq in seq_values:
            dict_seqn_repeating_col_vals[seq] = np.NaN
        # Dict that stores SEQN and its respective value in that column,
        # keeps updating the value while going through all same columns
        for column in columns:
            if a_repeating_col in column:
                # find the index of this column
                index = (list(df.columns)).index(column)
                col_values = list(df.iloc[:,index])
                # need to go through col_vals to add to dict
                for i in range(len(col_values)):
                    dict_key = seq_values[i]
                    # only replace it value isn't a NaN
                    if (col_values[i] == col_values[i]): #NaNs aren't equal to itself
                        dict_value = col_values[i]
                        dict_seqn_repeating_col_vals[dict_key] = dict_value

                cols_to_keep = [i for i in range(len(df.columns)) if i != index]
                df = df.iloc[:,cols_to_keep] 

        # Add a df containing this info to df
        seqn_list = list(dict_seqn_repeating_col_vals.keys())
        value_list = list(dict_seqn_repeating_col_vals.values())
        new_df = pd.DataFrame({'SEQN': seqn_list, a_repeating_col: value_list})
        df = df.merge(new_df, on='SEQN', how='outer', sort=True)

    return df

def filter_columns_SEQN(root_cats, year, threshold, filtered_SEQN, columns_dict):
    # Takes in a list of root categories, a year bracket, and a threshold 
    # for the % of NaN values in a column and returns a df with columns 
    # from all root cats with a % of NaN values less than the threshold
    # and filters SEQN 
    #root_loc = '/mnt/c/Users/alicia.mathew/Desktop/Project/NHANES data'

    # Empty Df for year to store all data
    year_df = pd.DataFrame({'SEQN': []})

    for root_cat in root_cats:
        data_dir = os.path.join(root_loc, 'diabetes_project/datasets', 'NHANES '+year, root_cat)

        # Empty Df for root_cat to store all data
        root_cat_df = pd.DataFrame({'SEQN': []})

        # Get a list of XPT files from data_dir
        xpt_list = locate_xpt_files(data_dir)

        xpt_loc_count = 0
        for xpt_loc in xpt_list:
            temp_df = pd.read_sas(xpt_loc) 
            if ('SEQN' in temp_df.columns):
                # Filter SEQN: 
                temp_df = temp_df.loc[temp_df['SEQN'].isin(filtered_SEQN)]

                # Filter columns:
                # List of columns in temp_df that are present across all years
                in_common = [column for column in temp_df.columns if \
                    column in columns_dict[root_cat]] 
                # Df with filtered columns
                temp_df = temp_df[in_common]

                xpt_loc_count += 1

                root_cat_df = root_cat_df.merge(temp_df, on='SEQN', how='outer',\
                    sort=True)
        
        print('There is a filtered dataframe of {} from {}'.format(root_cat, year))
        
        # Merge into YEAR dataframe before checking NaN values:
        year_df = year_df.merge(root_cat_df, on='SEQN', how='outer', sort=True)
        year_df = combine_same_columns(year_df)
    print('There is a filtered dataframe from {}'.format(year))
    
    if len(year_df) != 0:
        # Check NaN values:
        # List of columns with % of NaN values less than threshold
        under_threshold = []
        for column in year_df.columns:
            if year_df[column].isnull().sum()/len(year_df) < threshold:
                under_threshold.append(column)

        # Df for YEAR containing columns that meet the % NaN values criteria
        year_df = year_df[under_threshold]

    print('There is a filtered (by % NaN values) dataframe of {}'.format(year))
    return year_df

# %%

