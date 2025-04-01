# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:35:58 2024

@author: Judith
"""

# Import necessary packages
import os
import pandas as pd


# Read the .dat files into dataframes (stored in a dictionary)
def dat_to_df_dict(path):
    # List all .dat files in the directory
    dat_files = [f for f in os.listdir(path) if f.endswith(".dat")]
    
    # Create an empty dictionary to store dataframes
    dfs = {}
    
    # Loop through each .dat file and read it into a dataframe
    for i, file in enumerate(dat_files):
        # Construct the full file path
        file_path = os.path.join(path, file)
        
        # Read the .dat file into a dataframe
        df = pd.read_csv(file_path, delim_whitespace=True)
        
        # Use custom keys for the dataframes, e.g., data1, data2, data3, ...
        key = "data{}".format(i+1)
        dfs[key] = df
    return dfs

# Read the .txt files into dataframes (stored in a dictionary)
def txt_to_df_dict(path):
    # List all .dat files in the directory
    dat_files = [f for f in os.listdir(path) if f.endswith(".txt")]
    
    # Create an empty dictionary to store dataframes
    dfs = {}
    
    # Loop through each .dat file and read it into a dataframe
    for i, file in enumerate(dat_files):
        # Construct the full file path
        file_path = os.path.join(path, file)
        
        # Read the .dat file into a dataframe
        df = pd.read_csv(file_path) #, delim_whitespace=True
        
        # Use custom keys for the dataframes, e.g., data1, data2, data3, ...
        key = "data{}".format(i+1)
        dfs[key] = df
    return dfs

# calibrate CH4 measurements with the instruments Picarro G2301, Picarro G4302
# and Aeris (obtained from experiments in the lab by Carina van der Veen)
def calibrate(x, inst, typ):
        if inst == 'G23':
            if typ == 'CH4': 
                return 1.03127068196 * x - 0.15799666857 
            if typ == 'CO2':
                return 1.0088 * x + 0.320
        elif inst =='G43':
            if typ == 'CH4':
                methane = 1.01924906721 * x - 0.05887406866
                return methane
            elif typ == 'C2H6':
                ethane = 0.9950 * x            
                return ethane    
        elif inst == 'aer':
            if typ == 'CH4':
                return 1.01354227768 * x - 0.05055326961
            
# ...
def merge_with_gps(df_CH4,gps):
    df_merged = df_CH4.copy(deep=True)
    df_merged['time_round'] = df_merged.index.round('1s')
    df_merged = df_merged.join(gps,on='time_round')
    df_merged.drop(['time_round'],axis=1,inplace=True)
    return df_merged

# def merge_interpolate(df1,df2,col): # used for Rotterdam UU car
#         if df1.index.name == col:
#             combined = pd.merge(df1,df2,on=col,how='outer')
#             combined = combined.sort_values(by=col)
#             combined = combined.interpolate(method='linear')


def merge_interpolate_left(df1,df2,col):
    
    combined = pd.merge(df1,df2,on=col,how='left')
    combined = combined.sort_values(by=col)
    combined = combined.interpolate(method='linear')
    return combined

def combine_loc_and_rr(row):
    return (row['Loc'], row['Release_rate'])

def combine_city_loc_and_rr(row):
    return (row['City'], row['Loc'], row['Release_rate'])


def delete_duplicate_indices(df,name_index_column):
    """
    Only keep first occurence in case of duplicate indices.

    params:
    df (pandas dataframe): dataframe with potential duplicates
    name_index_column (string): name of the index column (default: 'index')
    
    returns: (dataframe)
    returns dataframe without duplicate indices
    """ 
    
    # Reset the index to make the index a column
    df_reset = df.reset_index()
    # Drop duplicates based on the original index column, keeping the first occurrence
    df_dedup = df_reset.drop_duplicates(subset=name_index_column, keep='first')
    # Set the index back to the original index column
    df_cleaned = df_dedup.set_index(name_index_column)
    # Optionally, sort the index if needed
    df_cleaned = df_cleaned.sort_index()
    
    return df_cleaned

