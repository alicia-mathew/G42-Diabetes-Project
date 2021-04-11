# %%

import pandas as pd
import numpy as np
import os
from implementation import locate_xpt_files
from implementation import intersection
from implementation import dict_SEQN

years = ['1999-2000','2001-2002','2003-2004','2005-2006','2007-2008','2009-2010',\
    '2011-2012','2013-2014','2015-2016','2017-2018']
root_cats = ['Demographics data', 'Dietary data', 'Examination data', 'Laboratory data', 'Questionnaire data']

dict_SEQN_across_root_cats = {}
for root_cat in root_cats:
    # dict of SEQN that stores a list of years that SEQN appears in 
    sequence_dict = dict_SEQN(years, root_cat)
    for seqn in sequence_dict:
        if seqn not in dict_SEQN_across_root_cats:
            dict_SEQN_across_root_cats[seqn] = sequence_dict[seqn]
        else:
            for year in sequence_dict[seqn]:
                if year not in dict_SEQN_across_root_cats[seqn]:
                    dict_SEQN_across_root_cats[seqn].append(year)
                    (dict_SEQN_across_root_cats[seqn]).sort()

#%%

# Contains all SEQN present in data and shows in which year brackets
num_years_present = {}
most_num_years = 0
for seqn in dict_SEQN_across_root_cats:
    the_years = dict_SEQN_across_root_cats[seqn]
    num_years_present[seqn] = len(the_years)
    if num_years_present[seqn] > most_num_years:
        most_num_years = num_years_present[seqn]

print('The most number of times a SEQN occurrs across all years: {}'.format(most_num_years))

# Find SEQN present in the most number of year brackets:
SEQN_in_most_num_years = dict(filter(lambda item: item[1] >= most_num_years, num_years_present.items()))

# Dict of all SEQN present in most year-brackets and which year-brackets:
SEQN_in_most_num_years2 = dict(filter(lambda item: item[0] in SEQN_in_most_num_years, \
    dict_SEQN_across_root_cats.items()))

#%%
## Find common columns in each root_cat across all years:
from implementation import find_common_columns

# Contains root_cat and its columns present across each year bracket
columns_across_years = {} 
for root_cat in root_cats:
    common_columns = find_common_columns(years, root_cat)
    columns_across_years[root_cat] = common_columns
    print('Finished {}'.format(root_cat))

for root_cat in root_cats:
    print('{} has {} common columns across all years.'.format(root_cat, \
        len(columns_across_years[root_cat])))

# %%
from implementation import combine_same_columns
from implementation import filter_columns_SEQN
from implementation import combine_same_columns

# Filters df for each YEAR by SEQN, common columns, and % of NaN values
root_cats = ['Demographics data', 'Dietary data', 'Examination data', 'Laboratory data', 'Questionnaire data']
test1999 = filter_columns_SEQN(root_cats, '1999-2000', 0.7, SEQN_in_most_num_years2, \
    columns_across_years)
print('##################################################')
test2001 = filter_columns_SEQN(root_cats, '2001-2002', 0.7, SEQN_in_most_num_years2, \
    columns_across_years)
print('##################################################')
#%%
test2003 = filter_columns_SEQN(root_cats, '2003-2004', 0.7, SEQN_in_most_num_years2, \
    columns_across_years)
print('##################################################')
#%%
test2005 = filter_columns_SEQN(root_cats, '2005-2006', 0.7, SEQN_in_most_num_years2, \
    columns_across_years)
print('##################################################')
#%%
test2007 = filter_columns_SEQN(root_cats, '2007-2008', 0.7, SEQN_in_most_num_years2, \
    columns_across_years)
print('##################################################')
test2009 = filter_columns_SEQN(root_cats, '2009-2010', 0.7, SEQN_in_most_num_years2, \
    columns_across_years)
print('##################################################')
#%%
test2011 = filter_columns_SEQN(root_cats, '2011-2012', 0.7, SEQN_in_most_num_years2, \
    columns_across_years)
#%%
print('##################################################')
test2013 = filter_columns_SEQN(root_cats, '2013-2014', 0.7, SEQN_in_most_num_years2, \
    columns_across_years)
print('##################################################')
test2015 = filter_columns_SEQN(root_cats, '2015-2016', 0.7, SEQN_in_most_num_years2, \
    columns_across_years)
print('##################################################')
test2017 = filter_columns_SEQN(root_cats, '2017-2018', 0.7, SEQN_in_most_num_years2, \
    columns_across_years)
# %%

## removed Physical Activity Monitor files in Examination data and
## Food Frequency Questionnaire - Output from DietCalc Software in Dietary data

