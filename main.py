import pandas as pd
import numpy as np
import os
from pathlib import Path
root_loc = os.path.abspath(".")

#  Some data files have multiple rows with data on the same patient and need to separate these
#  files from the rest in order to create one large dataframe for each year bracket.

# 1. Create a text file containing the names of all data files with multiple rows per patient (SEQN)
years = ['1999-2000','2001-2002','2003-2004','2005-2006','2007-2008','2009-2010',\
    '2011-2012','2013-2014','2015-2016','2017-2018']
root_cats = ['Demographics data', 'Dietary data', 'Examination data', \
    'Laboratory data', 'Questionnaire data']

from implementation_final import locate_files_with_repeated_seqn

# 1.1 Find all files with repeated SEQN and dump to file 'files_with_seqn_repeats.txt'
all_repeated_files = []
for root_cat in root_cats:
    repeated_files_in_root_cat = locate_files_with_repeated_seqn(years, root_cat)
    for xpt_file in repeated_files_in_root_cat:
        all_repeated_files.append(xpt_file)

with open("files_with_seqn_repeats.txt", "r") as f:
    files_with_repeated_seqn = f.read().split('\n')

# 2. Merge data for one year bracket using only data files that contain unique patients (SEQN)
from implementation_final import merge_data_for_one_year_bracket

for year in years:
    year_df = merge_data_for_one_year_bracket(root_cats, year, skip_files=files_with_repeated_seqn)
    year_df.to_pickle("{}_df.pkl".format(year))

# 3. Create "complete" dataframes for each year bracket including prescription and diet data, where 
# SEQN repeats
#    In this step, only certain Dietary files' data and Prescription meds data were included (among 
#    the list of files with repeated SEQN) in the "complete" dataframe
#    The other files contained data files that were either too large or that seemed "unnecessary"
#    to include because of their seeminly weak relation to diabetes
from implementation_final import updated_pres_meds_df, drug_class_names, diet_supp_df, complete_df

for year in years:
    print('Creating {} prescription medications df.'.format(year))
    pres_meds_df = updated_pres_meds_df(year, drug_class_names)
    pres_meds_df.to_pickle("{}_pres_meds_df.pkl".format(year))

for year in years:
    print('Creating {} dietary supplement df.'.format(year))
    dietary_supp_df = diet_supp_df(year)
    dietary_supp_df.to_pickle("{}_dietary_supp_df.pkl".format(year))

from implementation_final import complete_df

for year in years:
    print('Creating complete {} df.'.format(year))
    complete_year_df = complete_df(year)
    complete_year_df.to_pickle("{}_complete_df.pkl".format(year))

# 4. Create a master dataframe containing the first occurrence of each patient
#    in each year bracket's complete dataframe and then filter the columns by
#    getting rid of the ones with % NaN values that exceed a certain threshold
from implementation_final import create_master_df, filtered_columns_df

master_df = create_master_df(years)
master_df.to_pickle("master_df.pkl")

# Dataframe for Purely Data-driven Approach: using NaN value threshold = 0.5
master_df_with_filtered_cols = filtered_columns_df(master_df, 0.5)
master_df_with_filtered_cols.to_pickle('master_df_with_filtered_cols_50.0')

# Dataframe for Domain-driven Approach: using NaN value threshold = 0.55
final_df = filtered_columns_df(master_df, 0.55)
final_df.to_pickle('final_df')

################################################################################################################################################
#### DONE RUNNING FILE - ONCE ONLY #############################################################################################################