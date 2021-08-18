import pandas as pd
import numpy as np
import os
from pathlib import Path
root_loc = os.path.abspath(".")

# 6. Domain-driven Approach
print('############## WITH LAB DATA  #################\n')
print('############## DOMAIN-DRIVEN APPROACH #########')

df_file_path = os.path.join(root_loc, 'Aug5-dataframes/domain_driven/final_df')
final_df = pd.read_pickle(df_file_path)

# 6.1 Clean dataframe

# 6.1.1 Using Excel Sheet, only keep relevant columns
excel_path = os.path.join(root_loc, 'final_df_columns_analysis.xlsx')
xls = pd.ExcelFile(excel_path)
# Sheet5 has the relevant columns - 'dom' refers to 'domain-integrated'
relevant_cols_dom_df = pd.read_excel(xls, sheet_name='Sheet5')
# Only keep relevant columns
relevant_cols_dom_df.drop(relevant_cols_dom_df.columns[[0,1,3,4]], axis=1, inplace=True)
relevant_cols_dom_df['Feature_Col_Name'] = list(map(lambda s: s.split('\'')[1], list(relevant_cols_dom_df['Feature_Col_Name'])))
all_relevant_cols_dom = list(relevant_cols_dom_df['Feature_Col_Name'])
all_relevant_cols_dom.remove('BPQ090D') # isn't included in final_df
final_df = final_df[all_relevant_cols_dom]

# 6.1.2 Feature Engineering
from implementation_final import engineer_features

engineered_df = engineer_features(final_df)

# 6.1.3 Sort columns based on whether they are continuous, categorical, mixed or of object dtype
from implementation_final import cols_analysis_df

relevant_cols_dom_df = pd.merge(relevant_cols_dom_df, cols_analysis_df, on='Feature_Col_Name',\
    how='outer', sort=True)
relevant_cols_dom_df = relevant_cols_dom_df[relevant_cols_dom_df['Feature_Col_Name'].isin(all_relevant_cols_dom)]

cont_df_dom = relevant_cols_dom_df[relevant_cols_dom_df['Feature_Col_Dtype'] == 'CONTINUOUS']
cont_cols_dom = list(cont_df_dom['Feature_Col_Name'])
cat_df_dom = relevant_cols_dom_df[relevant_cols_dom_df['Feature_Col_Dtype'] == 'CAT']
cat_cols_dom = list(cat_df_dom['Feature_Col_Name'])
mixed_df_dom = relevant_cols_dom_df[relevant_cols_dom_df['Feature_Col_Dtype'] == 'MIX']
mixed_cols_dom = list(mixed_df_dom['Feature_Col_Name'])
object_df_dom = relevant_cols_dom_df[relevant_cols_dom_df['Feature_Col_Dtype'] == 'OBJECT']
object_cols_dom = list(object_df_dom['Feature_Col_Name'])

# 6.1.4 Add new engineered feature columns to list of categorical columns
cols_in_eng_not_in_rel_cols = [col for col in list(engineered_df.columns) if col not in all_relevant_cols_dom]
cat_cols_dom.extend(cols_in_eng_not_in_rel_cols)

# 6.1.5 Remove columns used to create new engineered features from respective list of columns
cols_in_rel_cols_not_in_eng_df = [col for col in all_relevant_cols_dom if col not in engineered_df]
for col in cols_in_rel_cols_not_in_eng_df:
    if col in cont_cols_dom:
        cont_cols_dom.remove(col)
    elif col in cat_cols_dom:
        cat_cols_dom.remove(col)
    elif col in mixed_cols_dom:
        mixed_cols_dom.remove(col)
    else:
        object_cols_dom.remove(col)
        
# 6.1.6 Convert mixed columns to categorical columns
from implementation_final import mixed_cols_to_cat_cols

engineered_df = mixed_cols_to_cat_cols(engineered_df, mixed_cols_dom)
cat_cols_dom.extend(mixed_cols_dom)

# 6.1.7 Convert all categorical columns to contain string values
from implementation_final import convert_cat_col_vals_to_str

engineered_df = convert_cat_col_vals_to_str(engineered_df, cat_cols_dom)

# 6.1.8 Assign Diabetes Class Labels
from implementation_final import assign_diabetes_class_labels

engineered_df = assign_diabetes_class_labels(engineered_df)

# remove columns used for classification from list of continuous and categorical cols to transform
cont_cols_dom.remove('LBXGLU')
cat_cols_dom.remove('DIQ010')

# 6.2
# like the BMC paper, we restrict dataset to patients with data between 1999-2014 and then later, 2003-2014
# (Data Release Numbers: 1 (1999-2000), 2 (2001-2002) ..., 8 (2013-2014))
year_brackets = list(map(lambda n: str(n), [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0]))
engineered_df = engineered_df[engineered_df['SDDSRVYR'].isin(year_brackets)]
print('\nBaseline dataframe restricted to patients with data between 1999-2014.\n')

# drop columns not considered as features in dataset and too highly correlated with outcome
cols_to_drop = ['SEQN','LBXGH','LBXSGL','PHAFSTHR','DIQ050','DIQ160','DIQ170','DIQ180','SDDSRVYR',\
    'AntiDiabetic_Agents', 'Num_Days_Taken_AntiDiabetic_Agents', 'WTSAF2YR']
engineered_df = engineered_df.drop(columns=[col for col in cols_to_drop if col in list(engineered_df.columns)])
for col in cols_to_drop:
    if col in cont_cols_dom:
        cont_cols_dom.remove(col)
    else:
        cat_cols_dom.remove(col)

diabetes_distribution = pd.DataFrame(engineered_df['Diabetes_Class_Label'].value_counts())
print('Number of diabetic patients: {} ({:.3f}%)'.format(diabetes_distribution.loc[1][0],\
    diabetes_distribution.loc[1][0]/len(engineered_df)*100))
print('Number of prediabetic patients: {} ({:.3f}%)'.format(diabetes_distribution.loc[2][0],\
    diabetes_distribution.loc[2][0]/len(engineered_df)*100))
print('Number of non-diabetic patients: {} ({:.3f}%)\n'.format(diabetes_distribution.loc[0][0],\
    diabetes_distribution.loc[0][0]/len(engineered_df)*100))

# 6.3 Prepare Data further for Machine Learning
X = engineered_df.drop(columns=['Diabetes_Class_Label'], axis=1)
X.to_pickle('domain_driven_X_before_transformation')
y = engineered_df['Diabetes_Class_Label']
