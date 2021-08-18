import pandas as pd
import numpy as np
import os
from pathlib import Path
root_loc = os.path.abspath(".")

# 5. Purely Data-Driven Approach
print('############## WITHOUT LAB DATA  ##############\n')
print('############## PURELY DATA-DRIVEN APPROACH ####\n')

df_file_path = os.path.join(root_loc, 'Aug5-dataframes/data_driven/master_df_with_filtered_cols_50.0')
master_df_with_filtered_cols = pd.read_pickle(df_file_path)

# 5.1 "Clean up" dataframe
# 5.1.1 Sort columns based on whether they are continuous, categorical, mixed or of object dtype
#     and drop redundant columns
from implementation_final import all_relevant_cols, all_continuous_cols, all_cat_cols, all_mixed_cols, all_object_cols

relevant_cols = [col for col in list(master_df_with_filtered_cols.columns) if col in all_relevant_cols]
master_df_with_filtered_cols = master_df_with_filtered_cols[relevant_cols]

cont_cols = [col for col in list(master_df_with_filtered_cols.columns) if col in all_continuous_cols]
cat_cols = [col for col in list(master_df_with_filtered_cols.columns) if col in all_cat_cols]
mixed_cols = [col for col in list(master_df_with_filtered_cols.columns) if col in all_mixed_cols]
object_cols = [col for col in list(master_df_with_filtered_cols.columns) if col in all_object_cols]

# even though this is a purely data-driven approach, we can make the informed decision to drop
# specific columns from the baseline dataframe due large number of categories present
# e.g. Columns that were of object dtype have a very large number of categories that will 
#      eventually need to be encoded using OneHotEncoder before passing it on to the ML 
#      model. With a many columns with the same property, the dataframe will become too 
#      large to handle.

# Drop object dtype cols
master_df_with_filtered_cols = master_df_with_filtered_cols.drop(columns=object_cols)

# 5.1.1.1 Convert mixed columns to categorical columns
from implementation_final import mixed_cols_to_cat_cols

master_df_with_filtered_cols = mixed_cols_to_cat_cols(master_df_with_filtered_cols, mixed_cols)
cat_cols.extend(mixed_cols)

# 5.1.1.2 Convert all categorical columns to contain string values
from implementation_final import convert_cat_col_vals_to_str

master_df_with_filtered_cols = convert_cat_col_vals_to_str(master_df_with_filtered_cols, cat_cols)

# 5.1.2 Assign Diabetes Class Labels
from implementation_final import assign_diabetes_class_labels

baseline_df = assign_diabetes_class_labels(master_df_with_filtered_cols)

# remove columns used for classification from list of continuous and categorical cols to transform
cont_cols.remove('LBXGLU')
cat_cols.remove('DIQ010')

# like the BMC paper, we restrict dataset to patients with data between 1999-2014 and then later, 2003-2014
# (Data Release Numbers: 1 (1999-2000), 2 (2001-2002), 3 (2003-2004), ..., 8 (2013-2014))
year_brackets = list(map(lambda n: str(n), [3.0,4.0,5.0,6.0,7.0,8.0]))
baseline_df = baseline_df[baseline_df['SDDSRVYR'].isin(year_brackets)]
print('\nBaseline dataframe restricted to patients with data between 2003-2014.\n')

# drop Lab data columns - columns with prefix 'LBX/LBD' are Blood tests and 'URX/URD' are Urine tests
lab_cols = [col for col in list(baseline_df.columns) if ((col[:3] == 'LBX') or (col[:3] == 'LBD')\
    or (col[:3] == 'URX') or (col[:3] == 'URD'))]

# drop columns not considered as features in dataset and too highly correlated with outcome
cols_to_drop = ['SEQN','PHAFSTHR','DIQ050','DIQ160','DIQ170','DIQ180','SDDSRVYR','PHDSESN']
cols_to_drop.extend(lab_cols)
baseline_df = baseline_df.drop(columns=[col for col in cols_to_drop if col in list(baseline_df.columns)])
for col in cols_to_drop:
    if col in cont_cols:
        cont_cols.remove(col)
    else:
        cat_cols.remove(col)

diabetes_distribution = pd.DataFrame(baseline_df['Diabetes_Class_Label'].value_counts())
print('Number of diabetic patients: {} ({:.3f}%)'.format(diabetes_distribution.loc[1][0],\
    diabetes_distribution.loc[1][0]/len(baseline_df)*100))
print('Number of prediabetic patients: {} ({:.3f}%)'.format(diabetes_distribution.loc[2][0],\
    diabetes_distribution.loc[2][0]/len(baseline_df)*100))
print('Number of non-diabetic patients: {} ({:.3f}%)\n'.format(diabetes_distribution.loc[0][0],\
    diabetes_distribution.loc[0][0]/len(baseline_df)*100))

# 5.2 Prepare Data further for Machine Learning
X = baseline_df.drop(columns=['Diabetes_Class_Label'], axis=1)
X.to_pickle('data_driven_X_before_transformation')
y = baseline_df['Diabetes_Class_Label']