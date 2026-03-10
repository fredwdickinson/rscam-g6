# ==================================
# Data cleaning modules collected all together.

import numpy as np
import pandas as pd 
import re

def clean_data(df, relevant_cols, early_age = 50):
    '''
    Collects all data cleaning functions into one pipeline, returning the clean data.
    '''

    clean_df = (df
            .pipe(groupby_age)
            .pipe(filter_columns, relevant_cols = relevant_cols)
            .pipe(remove_duplicates)
            .pipe(early_onset, early_age = early_age)
            )

    return clean_df

# ==================================

def groupby_age(df):
    ''' 
    Collects the different age columns (e.g. "Age At Diagnosis", "Diagnosis Age")
    into one column "Age". Drops the old columns.
    '''

    cols = np.array(df.columns.tolist())

    # \b is an empty space, sp .search() matches any " age " (or " Age ").
    pattern = re.compile(r'\bage\b', flags = re.IGNORECASE)
    age_cols = np.array([col for col in cols if pattern.search(col) and "Diagnosis" in col])

    # bfill takes the first nonempty entry across the age columns.
    df.loc[:, 'Age'] = df[age_cols].bfill(axis = 1).iloc[:, 0]
    df = df.drop(columns = age_cols)
    return df

# ==================================

def filter_columns(df, relevant_cols):
    ''' 
    Filters the data to only include specified relevant columns.
    Note that these must include "Sample ID" and "HGVSc" by default.
    '''

    # Strip the strings from the columns.
    # NOTE Incorporated the strip_strings() function that removes
    # leading and trailing whitespace only here.
    df.columns = df.columns.str.strip()

    # Check relevant columns includes the necessary list.
    necessary_cols = ["Sample ID", "HGVSc"]
    if set(relevant_cols).intersection(necessary_cols) == set(necessary_cols):
        # Add age on to the end.
        # NOTE This could just be added to necessary_cols ?
        good_cols = np.concatenate((relevant_cols, ['Age']))
        new_df = df[good_cols].copy()

        return new_df
    else:
        raise ValueError("Relevant columns must contain sample ID and HGVSc")
    
# ==================================

def remove_duplicates(df):
    ''' 
    Remove missing "Age" values, and drop duplicates that share
    the same Sample ID *AND* HGVSc information.
    NOTE this change after Meeting 2 w/ Xell.
    '''

    df = df.dropna(subset = ["Age"], axis = 0)
    df = df.drop_duplicates(subset = ["Sample ID", "HGVSc"], keep = "first")
    return df

# ==================================

def early_onset(df, early_age = 50):
    ''' 
    Create a new column that indicates if the cancer was early onset or not.
    NOTE Put this as default 50.
    '''

    df = df.copy()
    df["Early Onset"] = df["Age"] < early_age
    return df

# ==================================