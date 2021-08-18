import pandas as pd
import numpy as np
import os
from pathlib import Path
from pandas.core.arrays import categorical
root_loc = os.path.abspath(".")

# 1. Merge data for one year bracket using only data files that contain unique patients (SEQN)
def locate_xpt_files(loc:str)->list:
    """Takes the location of a folder and returns a list of .xpt files located anywhere
    within this folder and its subfolders.

    Args:
        loc (str): location of folder

    Returns:
        list: list of .xpt files
    """
    assert os.path.isdir(loc), "This function needs a valid direct, you \
        specified {}, check".format(loc)   
    flist = list(Path(loc).rglob("*.[xX][pP][tT]"))
    return flist

def array_unique(X:np.array)->bool:
    """Returns True if the array, X, contain unique elements.

    Args:
        X (np.array): array 

    Returns:
        bool: True if array has unique elements and False otherwise
    """
    return len(np.unique(X)) == len(X)

def locate_files_with_repeated_seqn(years:list, root_cat:str)->list:
    """Returns a list of filenames with repeating SEQN.

    Args:
        years (list): list of year brackets e.g. ['1999-2000','2001-2002',...]
        root_cat (str): root category e.g. 'Demographic data', 'Dietary data'...

    Returns:
        list: list of filenames
    """
    files_with_repeated_seqn = []

    for year in years:
        data_dir = os.path.join(root_loc, 'diabetes/NHANES data', 'NHANES '+year, root_cat)
        # Get a list of XPT files from data_dir
        xpt_list = locate_xpt_files(data_dir)
        # All unique SEQN in given year

        for xpt_loc in xpt_list:
            # Get list of SEQN in xpt_list
            print("Processing: {}".format(str(xpt_loc)))

            # Get file size
            # if file size > 100 MB
            # Load 10000 rows to check for duplicates at a time
            sas_rdr_obj = pd.read_sas(xpt_loc, chunksize=10000, iterator=True)

            for chunk in sas_rdr_obj:
                # Check if SEQN exists
                if 'SEQN' in chunk.columns:
                    # If yes, see if SEQN Unique
                    if not array_unique( np.array(list(chunk['SEQN']),dtype=np.int32) ):
                        files_with_repeated_seqn.append(str(xpt_loc))
                        break
                else:
                    break # If no, move on
    return files_with_repeated_seqn

def combine_cols_with_same_names_while_merging(df:pd.DataFrame)->pd.DataFrame:
    """Returns a df where all columns with the same prefix but different suffixes get
    combined into one.

    Args:
        df (pd.DataFrame): dataframe containing repeating columns with the same prefix
        but different suffixes.

    Returns:
        pd.DataFrame: dataframe where each column is unique
    """
    # Get column names of cols with _x, _y suffix
    cols_with_suffixes = [col for col in list(df.columns) if ('_x' in col or '_y' in col)]
    # Sort column names
    cols_with_suffixes.sort()

    new_df = df
    i = 0
    while i < len(cols_with_suffixes):
        # Get pair of columns: col_a and col_b
        col_a = cols_with_suffixes[i]
        col_b = cols_with_suffixes[i+1]
        # Locate rows in col_a that are null, but not_null in col_b
        rows_of_interest = ((df[col_a].isnull())) & (~(df[col_b].isnull()))
        # Overwrite these rows in col_a with values from col_b
        new_df[col_a][rows_of_interest] = new_df[col_b][rows_of_interest]
        new_df = new_df.drop(columns=[col_b])
        new_df = new_df.rename(columns={col_a: col_a[:-2]})
        i += 2
    return new_df

def merge_data_for_one_year_bracket(root_cats:list, year:str, skip_files=None)->pd.DataFrame:
    """Returns a dataframe containing data for one year bracket while only 
    including patients from files that don't have repeated patients.

    Args:
        root_cats (list): list of root categories e.g. ['Demographic data', 'Dietary data'...]
        year (str): year bracket e.g. '1999-2000', '2001-2002', ..., '2017-2018'
        skip_files ([list], optional): list of file names to skip over. Defaults to None.

    Returns:
        [pd.DataFrame]: dataframe for one year bracket
    """
    # Empty Df for year to store all data
    year_df = pd.DataFrame({'SEQN': []})
    for root_cat in root_cats:
        data_dir = os.path.join(root_loc, 'diabetes/NHANES data', 'NHANES '+year, root_cat)
        # Empty Df for root_cat to store all data
        root_cat_df = pd.DataFrame({'SEQN': []})
        # Get a list of XPT files from data_dir
        xpt_list = locate_xpt_files(data_dir)
        for xpt_loc in xpt_list:
            if str(xpt_loc) in skip_files:
                continue
            temp_df = pd.read_sas(xpt_loc) # Load data
            if ('SEQN' in temp_df.columns):
                root_cat_df = root_cat_df.merge(temp_df, on='SEQN', how='outer', sort=True)
                root_cat_df = combine_cols_with_same_names_while_merging(root_cat_df)
        print('There is a dataframe of {} from {}'.format(root_cat, year))
        year_df = year_df.merge(root_cat_df, on='SEQN', how='outer', sort=True)
        year_df = combine_cols_with_same_names_while_merging(year_df)
    print('There is a dataframe from {}'.format(year))
    return year_df

# 2. Creating complete dataframes for each year bracket, now including data from files
#    with repeated patients

# 2.1 Prescription Medication: Create dataframes for each year bracket
def drugs_used_by_SEQN(df:pd.DataFrame, seqn:int)->list:
    """Returns a list of the Generic Drugs Codes of the drugs used by SEQN from df.

    Args:
        df (pd.DataFrame): DataFrame of Prescription Medications file
        seqn (int): patient SEQN number

    Returns:
        list: List of generic drug codes taken by each SEQN
    """
    df_with_only_seqn = df[df['SEQN'] == seqn]
    drug_codes = list(df_with_only_seqn['RXDDRGID'])
    drug_codes_edited = [drug_code for drug_code in drug_codes if drug_code != '']
    return drug_codes_edited

drug_class_names = {'AntiInfectives': [1], 'Cardiovascular_Coag_Agents': [40, 81],\
    'Other_System_Agents': [57, 242, 87, 113, 122],\
    'Hormones_Modifiers': [97], 'Immunomodulators': [20, 254], 'Antihyperlipidemic_Agents': [358],\
        'AntiDiabetic_Agents': [358], 'Others': [133, 28, 105, 153, 218, 331, 115, 358]}

def drug_code_to_dummy_col(drug_code:str, drug_info_df:pd.DataFrame)->str:
    """Returns the dummy column that a drug code will fall under.

    Args:
        drug_code (str): Generic drug code of drug consumed by a SEQN.
        drug_info_df (pd.DataFrame): File containing Drug Information for a year bracket.

    Returns:
        str: Dummy column representing a collection of drugs.
    """
    drug_code_row = drug_info_df[drug_info_df['RXDDRGID'] == drug_code]

    first_level_cat_code = drug_code_row.iloc[0][3]
    second_level_cat_code =  drug_code_row.iloc[0][4]

    for drug_class in drug_class_names:
        list_first_level_cats = drug_class_names[drug_class]
        if first_level_cat_code == 358:
            if second_level_cat_code == 19:
                return 'Antihyperlipidemic_Agents' 
            elif second_level_cat_code == 99:
                return 'AntiDiabetic_Agents' 
            else:
                return 'Others'
        else:
            if first_level_cat_code in list_first_level_cats:
                return drug_class

def dict_dummy_cols_for_seqn(dict_drugs_seqn:dict, drug_info_df:pd.DataFrame)->dict:
    """Returns a dict where the keys are SEQN numbers and the values are lists of
    Drug Classes that the SEQN meds fall under.

    Args:
        dict_drugs_seqn (dict): Dict where the keys are SEQN numbers and the values 
        are list of drug codes of the meds hat SEQN takes
        drug_info_df (pd.DataFrame): File containing Drug Information for a year bracket.

    Returns:
        dict: dict of SEQN and list of Drug Classes
    """
    dict_dummy_cols_seqn = {}
    for seqn in dict_drugs_seqn:
        drugs_codes_used = dict_drugs_seqn[seqn]
        dummy_cols = []
        for drug_code in drugs_codes_used:
            dummy_col = drug_code_to_dummy_col(drug_code, drug_info_df)
            dummy_cols.append(dummy_col)
        dict_dummy_cols_seqn[seqn] = dummy_cols
    return dict_dummy_cols_seqn

def dict_num_pres_meds_for_seqn(df:pd.DataFrame)->dict:
    """Returns a dict of the number of prescription meds each SEQN takes.

    Args:
        df (pd.DataFrame): DataFrame of Prescription Medications file

    Returns:
        dict: dict of the number of prescription meds each SEQN takes.
    """
    dict_num_pres_meds = {}
    if 'RXD295' in df.columns:
        df_with_seqn_and_med_count = df[['SEQN', 'RXD295', 'RXDDRGID']]
        df_with_seqn_and_med_count = df_with_seqn_and_med_count[df_with_seqn_and_med_count['RXDDRGID'] != '']
        for seqn in df_with_seqn_and_med_count['SEQN'].unique():
            df_seqn = df_with_seqn_and_med_count[df_with_seqn_and_med_count['SEQN'] == seqn]
            med_count = df_seqn.iloc[0][1]
            if (seqn not in dict_num_pres_meds) and (not np.isnan(med_count)):
                dict_num_pres_meds[seqn] = med_count
        return dict_num_pres_meds
    else:
        # RXDCOUNT column present
        df_with_seqn_and_med_count = df[['SEQN', 'RXDCOUNT', 'RXDDRGID']]
        df_with_seqn_and_med_count = df_with_seqn_and_med_count[df_with_seqn_and_med_count['RXDDRGID'] != '']
        for seqn in df_with_seqn_and_med_count['SEQN'].unique():
            df_seqn = df_with_seqn_and_med_count[df_with_seqn_and_med_count['SEQN'] == seqn]
            med_count = df_seqn.iloc[0][1]
            if (seqn not in dict_num_pres_meds) and (not np.isnan(med_count)):
                dict_num_pres_meds[seqn] = med_count
        return dict_num_pres_meds

def dummy_col_df(dict_seqn_dummy_cols:dict)->pd.DataFrame:
    """Returns a DataFrame of Prescription Meds info including dummy columns.

    Args:
        dict_seqn_dummy_cols (dict): Dictionary of SEQN and their corresponding list
        of dummy columns.

    Returns:
        pd.DataFrame: DataFrame with dummy columns.
    """
    dict_drug_class_seqn = {'AntiInfectives': [], 'Cardiovascular_Coag_Agents': [], 'Other_System_Agents': [],\
        'Hormones_Modifiers': [],'Immunomodulators': [], 'Antihyperlipidemic_Agents': [], 'AntiDiabetic_Agents': [],\
        'Others': []}

    dict_seqn_dummy_cols_count = {}
    for seqn in dict_seqn_dummy_cols:
        counter = {drug_class: dict_seqn_dummy_cols[seqn].count(drug_class) \
            for drug_class in list(dict_drug_class_seqn.keys())}
        dict_seqn_dummy_cols_count[seqn] = counter
    
    for seqn in dict_seqn_dummy_cols_count:
        drug_class_counts = dict_seqn_dummy_cols_count[seqn]
        for drug_class in drug_class_counts:
            dict_drug_class_seqn[drug_class].append(drug_class_counts[drug_class])
    
    df = {'SEQN': list(dict_seqn_dummy_cols_count.keys())}
    for drug_class in dict_drug_class_seqn:
        df[drug_class] = dict_drug_class_seqn[drug_class]
    
    df = pd.DataFrame(df)
    return df

def num_days_drugs_taken_by_SEQN(df:pd.DataFrame, seqn:int)->dict:
    """Returns a dict of the drugs taken by each SEQN and how many days
    each were taken.

    Args:
        df (pd.DataFrame): DataFrame of Prescription Medications file
        seqn (int): patient SEQN number

    Returns:
        dict: dict of the drugs taken by each SEQN and how many days
        each were taken.
    """
    dict_num_days_drugs_taken = {}
    if 'RXD260' in df.columns:
        df_with_seqn_drug_and_days_taken = df[['SEQN', 'RXDDRGID', 'RXD260']]
        df_for_seqn = df_with_seqn_drug_and_days_taken[df_with_seqn_drug_and_days_taken['SEQN'] == seqn]
        for i in range(len(df_for_seqn)):
            drug_code = df_for_seqn.iloc[i][1]
            num_days_taken = df_for_seqn.iloc[i][2]
            # condition for 99999 / 77777 values:
            if (num_days_taken == 99999) or (num_days_taken == 77777):
                dict_num_days_drugs_taken[drug_code] = 0 # instead of np.NaN because then, when
                # summing, the result would be NaN
            else:
                dict_num_days_drugs_taken[drug_code] = num_days_taken
        return dict_num_days_drugs_taken
    else:
        # 'RXDDAYS' in columns:
        df_with_seqn_drug_and_days_taken = df[['SEQN', 'RXDDRGID', 'RXDDAYS']]
        df_for_seqn = df_with_seqn_drug_and_days_taken[df_with_seqn_drug_and_days_taken['SEQN'] == seqn]
        for i in range(len(df_for_seqn)):
            drug_code = df_for_seqn.iloc[i][1]
            num_days_taken = df_for_seqn.iloc[i][2]
            # condition for 99999 / 77777 values:
            if (num_days_taken == 99999) or (num_days_taken == 77777):
                dict_num_days_drugs_taken[drug_code] = 0 # instead of np.NaN because then, when
                # summing, the result would be NaN
            else:
                dict_num_days_drugs_taken[drug_code] = num_days_taken
        return dict_num_days_drugs_taken

def dict_num_days_drugs_taken_for_SEQN(df:pd.DataFrame)->dict:
    """Returns a dict of the drugs each SEQN takes and for how many days.

    Args:
        df (pd.DataFrame): DataFrame of Prescription Medications file

    Returns:
        dict: dict of the drugs each SEQN takes and for how many days.
    """ 
    dict_num_drugs_taken = {}
    if 'RXD260' in df.columns:
        df_with_seqn_and_num_day_drugs_taken = df[['SEQN', 'RXD260', 'RXDDRGID']]
        df_with_seqn_and_num_day_drugs_taken = df_with_seqn_and_num_day_drugs_taken[df_with_seqn_and_num_day_drugs_taken['RXDDRGID'] != '']
        for seqn in df_with_seqn_and_num_day_drugs_taken['SEQN'].unique():
            dict_drugs_and_days_taken = num_days_drugs_taken_by_SEQN(df_with_seqn_and_num_day_drugs_taken, seqn)
            if (seqn not in dict_num_drugs_taken): 
                dict_num_drugs_taken[seqn] = dict_drugs_and_days_taken
        return dict_num_drugs_taken
    else:
        # 'RXDDAYS' in columns:
        df_with_seqn_and_num_day_drugs_taken = df[['SEQN', 'RXDDAYS', 'RXDDRGID']]
        df_with_seqn_and_num_day_drugs_taken = df_with_seqn_and_num_day_drugs_taken[df_with_seqn_and_num_day_drugs_taken['RXDDRGID'] != '']
        for seqn in df_with_seqn_and_num_day_drugs_taken['SEQN'].unique():
            dict_drugs_and_days_taken = num_days_drugs_taken_by_SEQN(df_with_seqn_and_num_day_drugs_taken, seqn)
            if (seqn not in dict_num_drugs_taken): 
                dict_num_drugs_taken[seqn] = dict_drugs_and_days_taken
        return dict_num_drugs_taken

def dict_days_seqn_takes_drugs_in_dummy_col(dict_drugs_taken:dict, drug_info_df:pd.DataFrame)->dict:
    """Returns a dict with the drug codes replaced by their corresponding dummy 
    column names and sums up the total number of days the drugs under one dummy
    column names were taken. 

    Args:
        dict_drugs_taken (dict): dict of SEQN and the number of days each drug code was taken.
        drug_info_df (pd.DataFrame): File containing Drug Information for a year bracket.

    Returns:
        dict: dict with SEQN and the total number of days drugs in each dummy col were taken.
    """
    num_days_seqn_takes_drugs_in_dummy_col = {}
    for seqn in dict_drugs_taken:
        dict_num_days_taken = {'AntiInfectives': 0, 'Cardiovascular_Coag_Agents': 0,\
            'Other_System_Agents': 0,'Hormones_Modifiers': 0, 'Immunomodulators': 0,\
                'Antihyperlipidemic_Agents': 0, 'AntiDiabetic_Agents': 0, 'Others': 0}

        dict_num_days_each_seqn_takes_drug = dict_drugs_taken[seqn]
        for drug_code in dict_num_days_each_seqn_takes_drug:
            num_days_taken = dict_num_days_each_seqn_takes_drug[drug_code]
            dummy_col = drug_code_to_dummy_col(drug_code, drug_info_df)
            dict_num_days_taken[dummy_col] += num_days_taken
        num_days_seqn_takes_drugs_in_dummy_col[seqn] = dict_num_days_taken
    return num_days_seqn_takes_drugs_in_dummy_col

def num_days_dummy_col_meds_taken_df(dict_seqn_dummy_cols_num_days:dict)->pd.DataFrame:
    """Returns a dataframe with the columns representing the total number of days each SEQN
    has taken drugs in a dummy col drug class.

    Args:
        dict_seqn_dummy_cols_num_days (dict): dict with each SEQN and the number of days
        they take drugs in each dummy col drug class.

    Returns:
        pd.DataFrame: dataframe with the columns representing the total number of days each SEQN
        has taken drugs in a dummy col drug class.
    """
    dict_dummy_cols_seqn = {'Num_Days_Taken_AntiInfectives': [], 'Num_Days_Taken_Cardiovascular_Coag_Agents': [],\
        'Num_Days_Taken_Other_System_Agents': [], 'Num_Days_Taken_Hormones_Modifiers': [],'Num_Days_Taken_Immunomodulators': [],\
            'Num_Days_Taken_Antihyperlipidemic_Agents': [], 'Num_Days_Taken_AntiDiabetic_Agents': [],\
                'Num_Days_Taken_Others': []}
    
    for seqn in dict_seqn_dummy_cols_num_days:
        dict_for_seqn = dict_seqn_dummy_cols_num_days[seqn]
        for dummy_col in dict_for_seqn:
            num_days_taken = dict_for_seqn[dummy_col]
            for col_name in dict_dummy_cols_seqn:
                if dummy_col in col_name:
                    dict_dummy_cols_seqn[col_name].append(num_days_taken)
    
    df = {'SEQN': list(dict_seqn_dummy_cols_num_days.keys())}
    for num_days_dummy_col in dict_dummy_cols_seqn:
        df[num_days_dummy_col] = dict_dummy_cols_seqn[num_days_dummy_col]
    df = pd.DataFrame(df)
    
    return df

def updated_pres_meds_df(year:str)->pd.DataFrame:
    """Returns an updated dataframe of Prescription Medications

    Args:
        year (str): Year bracket (1999-2000, 2001-2002, ..., 2017-2018)

    Returns:
        pd.DataFrame: Dataframe of Prescription Medications
    """
    data_file = os.path.join(root_loc, 'diabetes/NHANES data/NHANES ' + year, 'Questionnaire data',\
        'Prescription Medications/Data file', year + '_Prescription Medications.XPT')
    drug_info_file = os.path.join(root_loc, 'diabetes/NHANES data/NHANES ' + year,'Questionnaire data',\
        'Prescription Medications - Drug Information/Data file',\
            year + '_Prescription Medications - Drug Information.xpt')
    data_df = pd.read_sas(data_file)
    drug_info_df = pd.read_sas(drug_info_file)
    
    # Make sure the drug codes are strings
    drug_codes = list(data_df['RXDDRGID'])
    drug_codes_edited = []
    for a_drug_code in drug_codes:
        string_code = str(a_drug_code)
        split = string_code.split('\'')
        drug_codes_edited.append(split[1])
    data_df['RXDDRGID'] = drug_codes_edited

    drug_codes = list(drug_info_df['RXDDRGID'])
    drug_codes_edited = []
    for a_drug_code in drug_codes:
        string_code = str(a_drug_code)
        split = string_code.split('\'')
        drug_codes_edited.append(split[1])
    drug_info_df['RXDDRGID'] = drug_codes_edited

    # Column for number of Prescription Meds (in total) taken per patient
    dict_seqn_num_pres_meds = dict_num_pres_meds_for_seqn(data_df) 

    dict_drugs_by_seqn = {} 
    for seqn in data_df['SEQN']:
        drugs_for_seqn = drugs_used_by_SEQN(data_df, seqn)
        if drugs_for_seqn != []:
            dict_drugs_by_seqn[seqn] = drugs_for_seqn
    
    # What Drug Classes dummy columns to fill for each SEQN
    dict_seqn_dummy_cols = dict_dummy_cols_for_seqn(dict_drugs_by_seqn, drug_info_df)
    # Dataframe with SEQN, and all dummy columns
    df_dummy_cols = dummy_col_df(dict_seqn_dummy_cols) 
    # Dataframe with SEQN, number of drugs taken in dummy cols, and total number of meds taken:
    df1 = pd.DataFrame({'SEQN': list(dict_drugs_by_seqn.keys()),\
        'PRES_MED_COUNT': list(dict_seqn_num_pres_meds.values())})
    df1 = df1.merge(df_dummy_cols, on='SEQN', how='outer', sort=True)

    # Total number of days drugs in a dummy col drug class taken:
    dict_num_days_med_taken_for_seqn = dict_num_days_drugs_taken_for_SEQN(data_df)
    dict_num_days_dummy_col_taken_for_seqn = \
        dict_days_seqn_takes_drugs_in_dummy_col(dict_num_days_med_taken_for_seqn, drug_info_df)
    df2 = num_days_dummy_col_meds_taken_df(dict_num_days_dummy_col_taken_for_seqn)
    
    df = pd.merge(df1, df2, on='SEQN', how='outer', sort=True)
    return df

# 2.2 Dietary Supplements: Create dataframes for each year bracket
def supp_for_SEQN(df:pd.DataFrame, seqn:int)->list:
    """Returns a list of the Dietary Supplement IDs of those used by
    SEQN from df.

    Args:
        df (pd.DataFrame): DataFrame of Dietary Supplement file
        seqn (int): patient SEQN number

    Returns:
        list: List of Dietary Supplement IDs taken by each SEQN
    """
    df_with_only_seqn = df[df['SEQN'] == seqn]
    supp_ids = list(df_with_only_seqn['DSDSUPID'])
    return supp_ids

def DSDPIDs_from_supp_IDs(list_supp_id:list, prod_info_df:pd.DataFrame)->int:
    """Returns a list of DSDPIDs of Dietary Supplement IDs.

    Args:
        list_supp_id (int): list of Dietary Supplement IDs
        prod_info_df (pd.DataFrame): Dietary Supplement Product Info dataframe.

    Returns:
        int: list of DSDPIDs
    """
    list_DSDPIDs = []
    for supp_id in list_supp_id:
        if ((str(supp_id)[0] == '1') and (not pd.isnull(supp_id))):
            supp_id_df = prod_info_df[prod_info_df['DSDSUPID'] == supp_id]
            dsdpid = supp_id_df['DSDPID'].iloc[0]
            list_DSDPIDs.append(dsdpid)
    #else:
        #return np.NaN
        return list_DSDPIDs

def DSPID_to_ing_cats(dsdpid:int, ing_info_df:pd.DataFrame)->dict:
    """Returns a dict showing the 5 ingredient categories the DSPID product contain
    ingredients in.

    Args:
        dsdpid (int): DSPID
        ing_info_df (pd.DataFrame): Dietary Supplement Ingredient Info dataframe.

    Returns:
        dict: dict showing the 5 ingredient categories the DSPID product contain
        ingredients in
    """
    dict_ing_cats = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    dsdpid_df = ing_info_df[ing_info_df['DSDPID'] == dsdpid]
    ing_cats = list(dsdpid_df['DSDCAT'].unique())
    for ing_cat in ing_cats:
        if ing_cat in dict_ing_cats.keys():
            dict_ing_cats[ing_cat] = 1
    return dict_ing_cats

def ing_cats_for_SEQN(list_DSDPIDs:list, ing_info_df:pd.DataFrame)->dict:
    """Returns a dict of ingredient categories under which the dietary
    supplements a SEQN takes falls.

    Args:
        list_DSDPIDs (list): list of supplement DSDPIDs for each SEQN
        ing_info_df (pd.DataFrame): Dietary Supplement Ingredient Info dataframe.

    Returns:
        dict: dict of ingredient categories under which the dietary
        supplements a SEQN takes falls.
    """
    SEQN_ing_cat = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    #all_dsdpids = {}
    new_list_DSDPIDs = [dsdpid for dsdpid in list_DSDPIDs if not pd.isnull(dsdpid)]
    for dsdpid in new_list_DSDPIDs:
        dict_dsdpid_ing_cats = DSPID_to_ing_cats(dsdpid, ing_info_df)
        for ing_cat in dict_dsdpid_ing_cats:
            if dict_dsdpid_ing_cats[ing_cat] == 1:
                SEQN_ing_cat[ing_cat] = 1
    return SEQN_ing_cat

d = os.path.join(root_loc, 'diabetes/NHANES data')
lst = locate_xpt_files(d)
# Get all file names in folder
lst = [str(s) for s in lst]
# Get only Dietary data file names in folder
lst = [s for s in lst if ('Dietary data' in s)]
# Get only Dietary Supplement Use file names in folder
lst = [s for s in lst if ('Dietary Supplement Use 30' in s)]
# Get only Individual Dietary Supplements file names in folder
file_paths = []
for s in lst:
    if ('File 2' in s) or ('Individual Dietary Supplements' in s):
        file_paths.append(s)

def diet_supp_df(year:str)->pd.DataFrame:
    """Returns an updated df containing Dietary Supplement Info for each SEQN.

    Args:
        year (str): year bracket ('1999-2000', ..., '2017-2018')

    Returns:
        pd.DataFrame: Dietary Supplement df.
    """
    for file_path in file_paths:
        if year in file_path:
            data_df = pd.read_sas(file_path)
    
    product_info_file = os.path.join(root_loc, 'diabetes/NHANES data/NHANES ' + year,\
        'Dietary data','Dietary Supplement Database - Product Information/Data file',\
            year + '_Dietary Supplement Database - Product Information.XPT')
    ingredient_info_file = os.path.join(root_loc, 'diabetes/NHANES data/NHANES ' + year,\
        'Dietary data','Dietary Supplement Database - Ingredient Information/Data file',\
            year + '_Dietary Supplement Database - Ingredient Information.XPT')
    product_info_df = pd.read_sas(product_info_file)
    ingredient_info_df = pd.read_sas(ingredient_info_file)

    if year != '2017-2018':
        supp_ids = list(data_df['DSDSUPID'])
        supp_ids_edited = []
        for supp_id in supp_ids:
            string_id = str(supp_id)
            split = string_id.split('\'')
            supp_ids_edited.append(int(split[1]))
        data_df['DSDSUPID'] = supp_ids_edited
        
        # for each SEQN, get a list of supplement IDs
        supp_for_each_SEQN = {}
        for seqn in data_df['SEQN'].unique():
            supp_for_each_SEQN[seqn] = supp_for_SEQN(data_df, seqn)

        # Make sure the Dietary Supplement IDs are ints
        supp_ids = list(product_info_df['DSDSUPID'])
        supp_ids_edited = []
        for supp_id in supp_ids:
            string_id = str(supp_id)
            split = string_id.split('\'')
            if split[1] != '':
                supp_ids_edited.append(int(split[1]))
            else:
                supp_ids_edited.append(np.NaN)
        product_info_df['DSDSUPID'] = supp_ids_edited

        # List of all DSDPID from list of Supplement IDs
        supp_DSDPIDs = {}
        for seqn in supp_for_each_SEQN:
            if supp_for_each_SEQN[seqn] == []:
                supp_for_each_SEQN.pop(seqn)
        for seqn in supp_for_each_SEQN:
            list_supp_id = supp_for_each_SEQN[seqn]
            dsdpids = DSDPIDs_from_supp_IDs(list_supp_id, product_info_df)
            if dsdpids != []:
                supp_DSDPIDs[seqn] = dsdpids
    else:
        # for each SEQN, get a list of supplement IDs
        supp_DSDPIDs = {}
        for seqn in data_df['SEQN'].unique():
            df_with_only_seqn = data_df[data_df['SEQN'] == seqn]
            supp_ids = list(df_with_only_seqn['DSDPID'])
            supp_ids = [supp_id for supp_id in supp_ids if (1 < supp_id and supp_id < 19767)]
            if supp_ids != []:
                supp_DSDPIDs[seqn] = supp_ids
    
    # Ingredient cats for each SEQN
    ing_cats_SEQN = {}
    for seqn in supp_DSDPIDs:
        list_supp_DSDPIDs = supp_DSDPIDs[seqn]
        seqn_ing_cats = ing_cats_for_SEQN(list_supp_DSDPIDs, ingredient_info_df)
        ing_cats_SEQN[seqn] = seqn_ing_cats

    columns = {'SEQN': list(ing_cats_SEQN.keys()), 'Ingredient1_Vitamin': [],\
        'Ingredient2_Mineral': [], 'Ingredient3_Botanical': [],\
                'Ingredient4_Other': [], 'Ingredient5_Amino_Acid': []}
    for seqn in ing_cats_SEQN:
        ing_cats = ing_cats_SEQN[seqn]
        for ing_cat in ing_cats:
            for column in columns.keys():
                if str(ing_cat) in column:
                    columns[column].append(ing_cats[ing_cat])
        
    df = pd.DataFrame(columns)
    return df    

def complete_df(year:str)->pd.DataFrame:
    """Returns the "complete" dataframe for a given year. This contains dataframes
    for every data file that does not have multiple rows for a SEQN as well as 
    two other separate dataframes containing information on patients' dietary
    supplements and prescription meds.

    Args:
        year (str): '1999-2000', '2001-2002', ..., '2017-2018'

    Returns:
        pd.DataFrame: complete df for a given year
    """
    df_without_rep_seqn_file = os.path.join(root_loc,\
        'dfs_without_repeated_seqn_updated', year + '_df.pkl')
    df_without_rep_seqn = pd.read_pickle(df_without_rep_seqn_file)
    df_dietary_supp_file = os.path.join(root_loc,\
        'dietary_supplements_dfs', year + '_dietary_supp_df.pkl')
    df_dietary_supp = pd.read_pickle(df_dietary_supp_file)
    df_pres_meds_file = os.path.join(root_loc,\
        'pres_meds_df', year + '_pres_meds_df.pkl')
    df_pres_meds = pd.read_pickle(df_pres_meds_file)

    df = df_without_rep_seqn.merge(df_dietary_supp, on='SEQN',\
        how='outer', sort=True)
    df = df.merge(df_pres_meds, on='SEQN', how='outer', sort=True)
    return df

# 3. Create a master dataframe containing the first occurrence of each patient
#    in each year bracket's complete dataframe
def combine_dfs(df1:pd.DataFrame, df2:pd.DataFrame)->pd.DataFrame:
    """Combines two dataframes of different dimensions.

    Args:
        df1 (pd.DataFrame): first dataframe with all SEQN
        df2 (pd.DataFrame): second dataframe with SEQN that aren't in df1

    Returns:
        pd.DataFrame: combined dataframe
    """
    # 1. Columns that are in both df1 and df2
    common_cols = [col for col in list(df1.columns) if col in list(df2.columns)]
    # 2. Create two dfs from df1 and df2 with only the common_cols
    df1_common_cols = df1[common_cols]
    df2_common_cols = df2[common_cols]
    # 3. Create a central df and let it be the concatenation of the above
    central_df = pd.concat([df1_common_cols, df2_common_cols], sort=False, ignore_index=True)
    common_cols.remove('SEQN')
    # 4. Create a df from df1 and df2 with the remaining columns that aren't common
    #    that we now need to merge
    df1_cols_to_merge = df1.drop(common_cols, axis = 1)
    df2_cols_to_merge = df2.drop(common_cols, axis = 1)
    # 5. Merge the dfs to the central_df
    central_df = central_df.merge(df1_cols_to_merge, on='SEQN',\
        how='outer', sort=True)
    central_df = central_df.merge(df2_cols_to_merge, on='SEQN',\
        how='outer', sort=True)
    return central_df

def all_seqn_filters_applied(df:pd.DataFrame)->pd.DataFrame:
    """Returns the dataframe after all SEQN filters are applied to it.

    Args:
        df (DataFrame): dataframe before any filters are apply to the SEQN.

    Returns:
        pd.DataFrame: dataframe after applying filters.
    """
    # 1. df that only stores SEQN whose diabetes label can be classified.
    df1_ind = ~(df['LBXGLU'].isnull())
    df2_ind = df['LBXGLU'].isnull() & df['DIQ010'] == 1
    df_ind = df1_ind | df2_ind
    # 2. Patient's Age >= 20
    df = df[df_ind & (df['RIDAGEYR'] >= 20)] 
    # 3. (Patient is not Female) OR (Patient is Female with negative Pregnancy Test)
    # instead of doing "not 1, not 3, and not 4"
    df = df[(df['RIAGENDR'] != 2) | ((df['RIAGENDR'] == 2) & (df['URXPREG'] == 2))]
    return df

def create_master_df(years:list)->pd.DataFrame:
    """Returns a master dataframe containing info for patients across
    all year brackets.

    Args:
        years (list): list of year brackets. e.g. ['1999-2000', '2001-2002',...]

    Returns:
        pd.DataFrame: master dataframe
    """
    main_df = pd.DataFrame({})
    for year in years:
        print('Processing year: {}'.format(year))
        # 1. Read complete df for year 
        year_file_path = os.path.join(root_loc, 'complete_year_dfs_updated',\
            year+'_complete_df.pkl')
        year_df = pd.read_pickle(year_file_path)
        # 2. Filter the seqn
        year_df_after_seqn_filters = all_seqn_filters_applied(year_df)

        if year == '1999-2000':
            main_df = year_df_after_seqn_filters
        # Create separate condition for these three year brackets since all SEQN are new, unique,
        # and without overlap
        elif year in ['2013-2014','2015-2016','2017-2018']:
            main_df = combine_dfs(main_df, year_df_after_seqn_filters)
        else:
            # Create a df with only SEQN that aren't already in the main_df
            seqn_to_append = [seqn for seqn in list(year_df_after_seqn_filters['SEQN'])
                if seqn not in list(main_df['SEQN'])]
            seqn_df = year_df_after_seqn_filters[year_df_after_seqn_filters['SEQN'].isin(seqn_to_append)]
            # Combine both dfs of different dimensions
            main_df = combine_dfs(main_df, seqn_df)
    return main_df
    
def filtered_columns_df(df:pd.DataFrame, threshold:int)->pd.DataFrame:
    """Returns a filtered df containing only variables available and continuous
    across all year brackets and with less than 50% missing values.

    Args:
        df (pd.DataFrame): dataframe
        threshold (int): threshold for highest % NaN values a column can have without
        getting dropped

    Returns:
        pd.DataFrame: filtered df containing fewer columns
    """
    # Specifies columns that are important - for filtering patients based on inclusion
    # criteria and for Diabetes Classification
    important_cols = ['SEQN','RIAGENDR','RIDAGEYR','LBXGLU','URXPREG','DIQ010','DIQ160']
    cols_to_keep = []
    for column in df.columns:
        if column in important_cols:
            cols_to_keep.append(column)
        else:
            # 3. Filters based on % of NaN vals in column
            if df[column].isnull().sum()/len(df) < threshold:
                cols_to_keep.append(column)
    condensed_df = df[cols_to_keep]
    return condensed_df

# 4. Assign Diabetes Class Label
def assign_diabetes_class_labels(df:pd.DataFrame)->pd.DataFrame:
    """Returns a df with an added column for the diabetes classification
    label using plasma glucose levels and the patient's answer to the
    question "Have you ever been told by a doctor that you have diabetes?"
    This is consistent with classification criteria used in the BMC article -
    could include other measures such as the column 'LBXGH' (HbA1c) 

    Args:
        df (pd.DataFrame): DataFrame with patients that fit the inclusion
        criteria.

    Returns:
        pd.DataFrame: DataFrame with patients that fit the inclusion criteria
        with an additional column for their diabetes classification label
    """
    # dataframe with the important columns used for classification
    df_with_imp_cols = df[['SEQN','LBXGLU','DIQ010']]
    diabetic = df_with_imp_cols[(df_with_imp_cols['LBXGLU'] >= 126) |\
        (df_with_imp_cols['DIQ010'] == 1)]
    prediabetic = df_with_imp_cols[(df_with_imp_cols['LBXGLU'] > 100) &\
        (df_with_imp_cols['LBXGLU'] < 126)]

    dict_of_seqn_and_class_label = {}
    for seqn in list(df['SEQN']):
        if seqn in list(diabetic['SEQN']):
            dict_of_seqn_and_class_label[seqn] = 1
        elif seqn in list(prediabetic['SEQN']):
            dict_of_seqn_and_class_label[seqn] = 2
        else:
            dict_of_seqn_and_class_label[seqn] = 0
    
    df['Diabetes_Class_Label'] = list(dict_of_seqn_and_class_label.values())
    # drop columns used for classification so they don't show up as being
    # very highly correlated with the target label
    df = df.drop(columns=['LBXGLU', 'DIQ010'])
    return df

# 5. Purely Data Driven Approach

# Convert all columns of object dtype to string dtype:
def convert_object_cols_to_str(df:pd.DataFrame, object_cols:list)->pd.DataFrame:
    """Returns a dataframe where the dtype of object columns is converted
    to string.

    Args:
        df (pd.DataFrame): DataFrame with object dtypes columns
        object_cols (list): Object dtype columns whose values are to be converted to strings

    Returns:
        pd.DataFrame: dataframe where object dtype columns are converted to string.
    """
    for col in object_cols:
        new_object_vals = []
        object_vals = list(df[col])
        for object_val in object_vals:
            if isinstance(object_val, float) or object_val == b'': # if object_val is a float (if its nan)
                new_object_vals.append(np.NaN)
            else:
                new_object_vals.append(str(object_val).split('\'')[1])
        df[col] = new_object_vals
    return df 

# 5.1 Create Excel sheet for analysis of columns and their properties
cols_analysis_file_path = os.path.join(root_loc, 'new_cols_analysis_df.xlsx')
cols_analysis_df = pd.read_excel(cols_analysis_file_path)
cols_analysis_df = cols_analysis_df.drop(cols_analysis_df.columns[0], axis=1)
# keep if Is_Relevant? is NaN or == 'Irrelevant'
relevant_cols_df1 = cols_analysis_df[cols_analysis_df['Is_Relevant?'].isnull()]
relevant_cols_df2 = cols_analysis_df[cols_analysis_df['Is_Relevant?'] == 'Irrelevant']
relevant_cols_df = pd.concat([relevant_cols_df1, relevant_cols_df2])
all_relevant_cols = list(relevant_cols_df['Feature_Col_Name'])
# group based on which features are continuous, categorical, and mixed
cont_df = relevant_cols_df[relevant_cols_df['Feature_Col_Dtype'] == 'CONTINUOUS']
all_continuous_cols = list(cont_df['Feature_Col_Name'])
cat_df = relevant_cols_df[relevant_cols_df['Feature_Col_Dtype'] == 'CAT']
all_cat_cols = list(cat_df['Feature_Col_Name'])
mixed_df = relevant_cols_df[relevant_cols_df['Feature_Col_Dtype'] == 'MIX']
all_mixed_cols = list(mixed_df['Feature_Col_Name'])
object_df = relevant_cols_df[relevant_cols_df['Feature_Col_Dtype'] == 'OBJECT']
all_object_cols = list(object_df['Feature_Col_Name'])

# 5.2 Categorical columns 
def convert_cat_col_vals_to_str(df:pd.DataFrame, cat_cols:list)->pd.DataFrame:
    """Returns a dataframe where the dtype of categorical columns is converted
    to string.

    Args:
        df (pd.DataFrame): DataFrame with categorical column values not being string
        cat_cols (list): Categorical columns whose values are to be converted to strings

    Returns:
        pd.DataFrame: dataframe where the dtype of categorical columns is converted
        to string.
    """
    for col in cat_cols:
        str_cat_vals = []
        cat_vals = list(df[col])
        for cat_val in cat_vals:
            if pd.isnull(cat_val):
                str_cat_vals.append(cat_val)
            else:
                str_cat_vals.append(str(cat_val))
        df[col] = str_cat_vals
    return df

# 5.2.1 Encode mixed columns and turn them into categorical columns - might need to add to this 
def mixed_cols_to_cat_cols(df:pd.DataFrame, mixed_cols:list)->pd.DataFrame:
    """Returns a dataframe after handling the columns with a mix of both continuous
    and categorical data. They will be transformed into columns with just categorical data
    depending on the ranges of the column values.
    e.g. if a value within the range 0-21 is stored as the value itself but every value 
         greater than 21, it encoded as 5555, the new column will store only 3 encoded values:
         0-10 => encoded as -1, 11-21 => encoded as 0, 21+ => encoded as 1.
    
    Args:
        df (pd.DataFrame): dataframe with continuous and categorical columns as
        well as some columns with a mix of both continuous and categorical data.
        mixed_cols (list): list of column names with mix of both continuous and categorical data.

    Returns:
        pd.DataFrame: dataframe with only continuous and categorical columns.
    """
    i = 0
    while i in range(len(mixed_cols)):
        mixed_col = mixed_cols[i]
        if mixed_col == 'PAD680':
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (0 <= val and val <= 400):
                    encoded_col_vals.append(-1)
                elif (401 <= val and val <= 800):
                    encoded_col_vals.append(0)
                elif (801 <= val and val <= 1200):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
        elif mixed_col == 'ALQ120Q':
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (0 <= val and val <= 120):
                    encoded_col_vals.append(-1)
                elif (121 <= val and val <= 241):
                    encoded_col_vals.append(0)
                elif (242 <= val and val <= 365):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
        elif mixed_col == 'SMD030':
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (6 <= val and val <= 79):
                    encoded_col_vals.append(-1)
                elif (val == 0):
                    encoded_col_vals.append(0)
                elif (80 <= val):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
        elif mixed_col == 'OCQ180':
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (0 <= val and val <= 40):
                    encoded_col_vals.append(-1)
                elif (41 <= val and val <= 80):
                    encoded_col_vals.append(0)
                elif (81 <= val and val <= 120):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
        elif mixed_col in ['HSQ470','HSQ480','HSQ490']:
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (0 <= val and val <= 10):
                    encoded_col_vals.append(-1)
                elif (11 <= val and val <= 20):
                    encoded_col_vals.append(0)
                elif (21 <= val and val <= 30):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
        elif mixed_col == 'INDFMMPI':
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (0 <= val and val <= 2.5):
                    encoded_col_vals.append(-1)
                elif (2.5 < val and val <= 5):
                    encoded_col_vals.append(0)
                elif (5 < val):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
        elif mixed_col == 'SLD010H':
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (2 <= val and val <= 6):
                    encoded_col_vals.append(-1)
                elif (7 <= val and val <= 11):
                    encoded_col_vals.append(0)
                elif (12 <= val):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
        elif mixed_col == 'DMDHRAGE':
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (18 <= val and val <= 45):
                    encoded_col_vals.append(-1)
                elif (46 <= val and val <= 79):
                    encoded_col_vals.append(0)
                elif (80 <= val):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
        elif mixed_col in ['DBD895','DBD900']:
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (0 <= val and val <= 10):
                    encoded_col_vals.append(-1)
                elif (11 <= val and val <= 20):
                    encoded_col_vals.append(0)
                elif (21 <= val):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
        elif mixed_col == 'DBD905':
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (0 <= val and val <= 50):
                    encoded_col_vals.append(-1)
                elif (51 <= val and val <= 100):
                    encoded_col_vals.append(0)
                elif (101 <= val <= 150):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
        elif mixed_col == 'DBD910':
            encoded_col_vals = []
            col_values = list(df[mixed_col])
            for val in col_values:
                if (0 <= val and val <= 60):
                    encoded_col_vals.append(-1)
                elif (61 <= val and val <= 120):
                    encoded_col_vals.append(0)
                elif (121 <= val <= 180):
                    encoded_col_vals.append(1)
                else:
                    encoded_col_vals.append(np.NaN)
            df[mixed_col] = encoded_col_vals
            i += 1
    return df

# 6. Create Transformation Pipelines for ML
# 6.1 Create baseline numerical and categorical pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

baseline_cont_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

baseline_cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('one_hot_encoder', OneHotEncoder()),
])

# 6.2 Create other transformation pipelines to test algorithm - RFC
from sklearn.impute import KNNImputer

# 6.2.1 Outline continuous and categorical imputers
knn_imp_cat = KNNImputer(n_neighbors=1)
knn_imp_cont = KNNImputer(n_neighbors=3)

cont_pipeline1 = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('standard scaler', StandardScaler()),
])

cat_pipeline1 = Pipeline([
    ('imputer', knn_imp_cat),
    ('one_hot_encoder', OneHotEncoder()),
])

cont_pipeline2 = Pipeline([
    ('imputer', knn_imp_cont),
    ('standard scaler', StandardScaler()),
])

cat_pipeline2 = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('one_hot_encoder', OneHotEncoder()),
])

cont_pipeline3 = Pipeline([
    ('imputer', knn_imp_cont),
    ('standard scaler', StandardScaler()),
])

cat_pipeline3 = Pipeline([
    ('imputer', knn_imp_cat),
    ('one_hot_encoder', OneHotEncoder()),
])

# ####################################################################################################

# 7. "Domain-Driven" Approach - Feature Engineering
def sexual_history(x:pd.core.series.Series)->int:
    """Returns a categorical value based on the sexual history feature
    column values of a patient row, x, in a dataframe.

    Args:
        x (pd.core.series.Series): a patient row in a dataframe

    Returns:
        int: categorical value resulting from the sexual history
        feature column values of a patient row.
    """
    df = x[['SXQ260','SXQ265','SXQ270','SXQ272']].copy()
    if (df==1).any():
        return 1
    elif (df==2).all():
        return 2
        
def hsv_status(x:pd.core.series.Series)->int:
    """Returns a categorical value based on the Herpes Simplex Virus status
    column values of a patient row, x, in a dataframe.

    Args:
        x (pd.core.series.Series): a patient row in a dataframe

    Returns:
        int: categorical value resulting from the Herpes Simplex Virus status
        feature column values of a patient row.
    """
    if x.LBXHE1 == 1 and x.LBXHE2 != 1:
        return 1
    if x.LBXHE1 != 1 and x.LBXHE2 == 1:
        return 2
    if x.LBXHE1 == 1 and x.LBXHE2 == 1:
        return 3
    if x.LBXHE1 == 2 and x.LBXHE2 == 2:
        return 4
        
def hepE_status(x:pd.core.series.Series)->int:
    """Returns a categorical value based on the Hepatitus E status
    column values of a patient row, x, in a dataframe.

    Args:
        x (pd.core.series.Series): a patient row in a dataframe

    Returns:
        int: categorical value resulting from the Hepatitus E status
        feature column values of a patient row.
    """

    if x.LBDHEM == 1:
        return 1
    elif x.LBDHEM == 2:
        if x.LBDHEG == 1:
            return 2
        elif x.LBDHEG == 2:
            return 3

def hepB_status(x:pd.core.series.Series)->int:
    """Returns a categorical value based on the Hepatitus B status
    column values of a patient row, x, in a dataframe.

    Args:
        xx (pd.core.series.Series): a patient row in a dataframe

    Returns:
        int: categorical value resulting from the Hepatitus B status
        feature column values of a patient row.
    """
    if x.LBDHBG == 1:
        return 1
    elif x.LBDHBG == 2:
        if x.LBXHBS == 1:
            if x.LBXHBC == 1:
                return 2
            if x.LBXHBC == 2:
                return 3
        elif x.LBXHBS == 2:
            if x.LBXHBC == 1:
                return 4
            if x.LBXHBC == 2:
                return 5
            
def sun_exposure(x:pd.core.series.Series)->int:
    """Returns a categorical value based on the sun exposure
    column values of a patient row, x, in a dataframe.
    Args:
        x (pd.core.series.Series): a patient row in a dataframe

    Returns:
        int: categorical value resulting from the sun exposure
        feature column values of a patient row.
    """
    if (x[['DEQ034A','DEQ034C','DEQ034D']]).notna().all():
        if x.DEQ034A <= 6:
            x.DEQ034A = 6 - x.DEQ034A
            value = round(x[['DEQ034A','DEQ034C','DEQ034D']].mean())
            if value > 5:
                value = 5
            return value

def resp_health(x:pd.core.series.Series)->int:
    """Returns a categorical value based on the respiratory health
    column values of a patient row, x, in a dataframe.

    Args:
        x (pd.core.series.Series): a patient row in a dataframe

    Returns:
        int: categorical value resulting from the respiratory health
        feature column values of a patient row.
    """
    df = x[['RDQ050','RDQ070','RDQ140']].copy()
    if (df==1).any():
        return 1
    elif (df==2).all():
        return 2
        
def phq_score(x:pd.core.series.Series)->int:
    """Returns a categorical value based on the depression
    column values of a patient row, x, in a dataframe.

    Args:
        x (pd.core.series.Series): a patient row in a dataframe

    Returns:
        int: categorical value resulting from the depression
        feature column values of a patient row.
    """
    phq = x[['DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090']].round()
    phq = phq.replace(7, np.nan)
    phq = phq.replace(9, np.nan)
    if phq.isna().sum() == 9:
        return np.nan
    else: 
        phq.fillna(phq.mode()[0], inplace=True)
        if phq.sum() < 5:
            return 1
        elif phq.sum() < 10:
            return 2
        elif phq.sum() < 15:
            return 3
        elif phq.sum() < 20:
            return 4
        elif phq.sum() <= 27:
            return 5
           
def fracture_history(x:pd.core.series.Series)->int:
    """Returns a categorical value based on the fracture history
    column values of a patient row, x, in a dataframe.

    Args:
        x (pd.core.series.Series): a patient row in a dataframe

    Returns:
        int: categorical value resulting from the fracture history
        feature column values of a patient row.
    """
    df = x[['OSQ010A','OSQ010B','OSQ010C']].copy()
    if (df==1).any():
        return 1
    elif (df==2).all():
        return 2

def urine_incont(x:pd.core.series.Series)->int:
    """Returns a categorical value based on the urology
    column values of a patient row, x, in a dataframe.

    Args:
        x (pd.core.series.Series): a patient row in a dataframe

    Returns:
        int: categorical value resulting from the urology
        feature column values of a patient row.
    """
    df = x[['KIQ005','KIQ042','KIQ044','KIQ046']].copy()
    df = df.replace(7, np.nan)
    df = df.replace(9, np.nan)
    if df.KIQ005 >= 2:
        df.KIQ005 = 2
    df.KIQ005 = 3 - df.KIQ005
    if (df==1).any():
        return 1
    elif (df==2).all():
        return 2

def engineer_features(df:pd.DataFrame)->pd.DataFrame:
    """Returns a dataframe with engineered features. Columns are dropped
    and engineered columns are added to represent the columns that have
    been dropped.

    Args:
        df (pd.DataFrame): dataframe that has feature columns that need
        to be engineered.

    Returns:
        [pd.DataFrame]: data with the engineered feature columns.
    """
    
    sexual_series = df.apply(lambda row: sexual_history(row), axis=1)
    print("Adding sexual history variable - EFSEXD")
    df['EFSEXD'] = sexual_series
    print("Removing sexual history variables - ['SXQ260','SXQ265','SXQ270','SXQ272']")
    df.drop(['SXQ260','SXQ265','SXQ270','SXQ272'], axis=1, inplace=True)
    
    herpes_series = df.apply(lambda row: hsv_status(row), axis=1)
    print("Adding herpes history variable - EFHSVI")
    df['EFHSVI'] = herpes_series
    print("Removing herpes history variables - ['LBXHE1','LBXHE2']")
    df.drop(['LBXHE1','LBXHE2'], axis=1, inplace=True)
    
    hepE_series = df.apply(lambda row: hepE_status(row), axis=1)
    print("Adding HepE history variable - EFHEPE")
    df['EFHEPE'] = hepE_series
    print("Removing hepE history variables - ['LBDHEM','LBDHEG']")
    df.drop(['LBDHEM','LBDHEG'], axis=1, inplace=True)
    
    hepB_series = df.apply(lambda row: hepB_status(row), axis=1)
    print("Adding HepB history variable - EFHEPB")
    df['EFHEPB'] = hepB_series
    print("Removing hepB history variables - ['LBXHBS','LBXHBC', 'LBDHBG']")
    df.drop(['LBXHBS','LBXHBC', 'LBDHBG'], axis=1, inplace=True)
    
    sun_series = df.apply(lambda row: sun_exposure(row), axis=1)
    print("Adding sun exposure variable - EFESUN")
    df['EFESUN'] = sun_series
    print("Removing sun exposure variables - ['DEQ034A','DEQ034C','DEQ034D']")
    df.drop(['DEQ034A','DEQ034C','DEQ034D'], axis=1, inplace=True)
    
    resp_series = df.apply(lambda row: resp_health(row), axis=1)
    print("Adding respiratory health variable - EFRESP")
    df['EFRESP'] = resp_series
    print("Removing respiratory health variables - ['RDQ050','RDQ070','RDQ140']")
    df.drop(['RDQ050','RDQ070','RDQ140'], axis=1, inplace=True)
    
    phq_series = df.apply(lambda row: phq_score(row), axis=1)
    print("Adding depression variable - EFPHQS")
    df['EFPHQS'] = phq_series
    print("Removing depression variables - ['DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090']")
    df.drop(['DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090'], axis=1, inplace=True)
    
    fracture_series = df.apply(lambda row: fracture_history(row), axis=1)
    print("Adding fracture history variable - EFFRAC")
    df['EFFRAC'] = fracture_series
    print("Removing fracture history variables - ['OSQ010A','OSQ010B','OSQ010C']")
    df.drop(['OSQ010A','OSQ010B','OSQ010C'], axis=1, inplace=True)
    
    urine_series = df.apply(lambda row: urine_incont(row), axis=1)
    print("Adding urine incontinence history variable - EFURIN")
    df['EFURIN'] = urine_series
    print("Removing urine incontinence history variables - ['KIQ005','KIQ042','KIQ044','KIQ046']")
    df.drop(['KIQ005','KIQ042','KIQ044','KIQ046'], axis=1, inplace=True)
       
    return df
