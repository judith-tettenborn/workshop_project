# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:35:33 2024

@author: Judith


# =============================================================================
# =============================================================================
# # Pre ANALYSIS FOR PEAKS THAT PASSED QC
# =============================================================================
# =============================================================================

 Calculate Area and main features important for later analysis



"""

#%% Load Packages & Data

# Modify Python Path Programmatically -> To include the directory containing the src folder
import sys
from pathlib import Path
import os
from peak_analysis.find_analyze_peaks import *

# V1 - works but hard-coded
# sys.path.append('C:\\Users\\Judit\\OneDrive - Universiteit Utrecht\\02_Coding\\Workshop_Best-practices-writing-code\\src')
# In Python, the "Python path" refers to the list of directories where Python looks for modules
# and packages when you try to import them in your scripts or interactive sessions. This path 
# is stored in the sys.path list. When you execute an import statement in Python, it searches 
# for the module or package you're trying to import in the directories listed in sys.path. 
# If the directory containing the module or package is not included in sys.path, Python won't 
# be able to find and import it.

# READ IN DATA
# path_fig       =  'C:/Users/Judit/OneDrive - Universiteit Utrecht/02_Coding\Workshop_Best-practices-writing-code/Figures/'
# path_procdata  = 'C:/Users/Judit/OneDrive - Universiteit Utrecht/02_Coding\Workshop_Best-practices-writing-code/Data/processed/'
# path_finaldata = 'C:/Users/Judit/OneDrive - Universiteit Utrecht/02_Coding\Workshop_Best-practices-writing-code/Data/final/'

# V2 - relative paths
ROOT_DIR = Path(os.path.abspath("")) #.parent  # Moves up one level
print(ROOT_DIR)

# Get the root directory (assuming the script is in the 'src' folder)
ROOT_DIR = Path(__file__).resolve().parent.parent  # Moves up from 'src' to project root




# Modify Python Path to include the 'src' folder
sys.path.append(str(ROOT_DIR / "src"))

# Define paths relative to the project root
path_fig       = ROOT_DIR / "results/figures"
path_procdata  = ROOT_DIR / "data" / "processed"
path_finaldata = ROOT_DIR / "data" / "final"

#-----

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from peak_analysis.find_analyze_peaks import *
from plotting.general_plots import *
#from helper_functions.data_handling import *
# functions used
# plot_CH4timeseries
# analyse_peak
# plot_indivpeaks_afterQC
# add_distance_to_df


# READ IN DATA
path_fig       =  'C:/Users/Judit/OneDrive - Universiteit Utrecht/02_Coding\Workshop_Best-practices-writing-code/Figures/'
path_procdata  = 'C:/Users/Judit/OneDrive - Universiteit Utrecht/02_Coding\Workshop_Best-practices-writing-code/Data/processed/'
path_finaldata = 'C:/Users/Judit/OneDrive - Universiteit Utrecht/02_Coding\Workshop_Best-practices-writing-code/Data/final/'

# -----------------------------------------------------------------------------
# Read in datafiles containing identified and quality-checked methane plumes
# -----------------------------------------------------------------------------
# Each row corresponds to a peak identified with the scipy function find_peaks, the index gives the datetime 
# of the maximum pint of the peak. In the quality check (QC) the peak were manually checked for validity,
# based on vehicle speed (e.g. car standing), instrument failures, distance to source, availability of GPS information... 
# The columns with the respective 'CH4 instrument name' (e.g. G2301 or Aeris), 'GPS', 'Loc', 'QC' are indicative
# if the peak is valid (if it is not valid one or more of those columns contain a 0 for this specific peak). 
# In the following script the peaks which are valid are further analysed (e.g. calculation of integrated area).

corrected_peaks_U       = pd.read_excel(path_procdata+"U_output_QC.xlsx",sheet_name='G4302',index_col='Datetime')
corrected_peaks_U3      = pd.read_excel(path_procdata+"U3_output_2perc_QC.xlsx",sheet_name='G2301',index_col='Datetime')
corrected_peaks_R_mUU   = pd.read_excel(path_procdata+"R_output_morningUU_QC.xlsx",sheet_name='G4302',index_col='Datetime')
corrected_peaks_R_mTNO  = pd.read_excel(path_procdata+"R_output_morningTNO_QC.xlsx",sheet_name='miro',index_col='Datetime')
corrected_peaks_R_aTNO  = pd.read_excel(path_procdata+"R_output_afternoon_QC.xlsx",sheet_name='miro',index_col='Datetime')


# Drop columns which are not necessary for the analysis
# 'Sure?' and 'corrected' contain comments on the quality check (e.g. why a peak was regarded, if a peak was kept, but there is uncertainty about its validity)
corrected_peaks_U       = corrected_peaks_U.drop(columns={'Peakstart','Peakend','corrected','Sure?','Overlap?','Car_passed? (Both)','Car_passed? (G4)'})
corrected_peaks_U3      = corrected_peaks_U3.drop(columns={'Peakstart','Peakend','corrected','Sure?'})
corrected_peaks_R_mUU   = corrected_peaks_R_mUU.drop(columns={'Peakstart','Peakend','corrected','Peak_old', 'Sure?'})
corrected_peaks_R_aTNO  = corrected_peaks_R_aTNO.drop(columns={'Peakstart','Peakend','Peakend_peakfinder','corrected','Sure?','Distance','CH4_ele_miro_TNO'})
corrected_peaks_R_mTNO  = corrected_peaks_R_mTNO.drop(columns={'Peakstart','Peakend','corrected','Distance','CH4_ele_miro_TNO'})



# -----------------------------------------------------------------------------
# Read in datafiles containing methane timeseries
# -----------------------------------------------------------------------------
# Utrecht
U_G4302 = pd.read_excel(path_procdata+'U_G23andG43.xlsx',sheet_name='G4302', index_col='Datetime')  
U_G2301 = pd.read_excel(path_procdata+'U_G23andG43.xlsx',sheet_name='G2301', index_col='Datetime')  
# Utrecht III
U3_G2301 = pd.read_csv(path_procdata+'U3_G2301.csv', index_col='Datetime', parse_dates=['Datetime'])
U3_aeris = pd.read_csv(path_procdata+'U3_aeris.csv', index_col='Datetime', parse_dates=['Datetime']) 

# Rotterdam
R_G4302 = pd.read_csv(path_procdata+'R_G4302.csv', index_col='Datetime', parse_dates=['Datetime'])  
R_G2301 = pd.read_csv(path_procdata+'R_G2301.csv', index_col='Datetime', parse_dates=['Datetime'])
R_aeris = pd.read_csv(path_procdata+'R_aeris.csv', index_col='Datetime', parse_dates=['Datetime'])  
R_miro = pd.read_csv(path_procdata+'R_miro.csv', index_col='Datetime', parse_dates=['Datetime'])
R_aerodyne = pd.read_csv(path_procdata+'R_aerodyne.csv', index_col='Datetime', parse_dates=['Datetime'])



# -----------------------------------------------------------------------------
# Define dictionaries containing informations about the dataset
# -----------------------------------------------------------------------------

# Utrecht
U_vars_G43 = {'df': U_G4302,
                'CH4col':  'CH4_ele_G43', 
                'spec':    'G43',
                'title':   'G4302',
                'city':    'Utrecht',
                'day':      'Day1'
                 }
U_vars_G23 = {'df': U_G2301,
                'CH4col':  'CH4_ele_G23',
                'spec':    'G23',
                'title':   'G2301',
                'city':    'Utrecht',
                'day':      'Day1'
                }

# Utrecht III
U3_vars_aeris = {'df': U3_aeris,
                'CH4col':  'CH4_ele_aeris', 
                'spec':    'aeris',
                'title':   'Aeris',
                'city':    'Utrecht_III',
                'day':      'Day1'
                 }
U3_vars_G23 = {'df': U3_G2301,
                'CH4col':  'CH4_ele_G23',
                'spec':    'G23',
                'title':   'G2301',
                'city':    'Utrecht_III',
                'day':      'Day1'
                }

# Rotterdam
R_vars_G43 = {'df': R_G4302,
                'CH4col':  'CH4_ele_G43', 
                'spec':    'G43',
                'title':   'G4302',
                'city':    'Rotterdam',
                'day':      'Day1'
                 }
R_vars_G23 = {'df': R_G2301,
                 'CH4col':  'CH4_ele_G23', 
                 'spec':    'G23',
                 'title':   'G2301',
                 'city':    'Rotterdam',
                 'day':      'Day1'
                 }
R_vars_aeris = {'df': R_aeris,
                'CH4col':  'CH4_ele_aeris', 
                'spec':    'aeris',
                'title':   'Aeris',
                'city':    'Rotterdam',
                'day':      'Day1'
                 }
R_vars_miro = {'df': R_miro,
                 'CH4col':  'CH4_ele_miro', 
                 'spec':    'miro',
                 'title':   'Miro',
                 'city':    'Rotterdam',
                 'day':      'Day1'
                 }
R_vars_aerodyne = {'df': R_aerodyne,
                 'CH4col':  'CH4_ele_aero', 
                 'spec':    'aero',
                 'title':   'Aerodyne',
                 'city':    'Rotterdam',
                 'day':      'Day1'
                 }


# Dictionary defining a color for each instrument (for plotting)

dict_color_instr = {'G2301': '#fb7b50',
              'G4302': '#00698b',
              'Aeris': '#91e1d7',
              'Miro': '#ffc04c',
              'Aerodyne': '#00b571',
              }


#%% Timeseries Overview

#from plotting.general_plots import *


# Utrecht -----------------------
path_savefig = path_fig + 'Utrecht_III/'
U3_G2301.name = 'G2301'
U3_aeris.name = 'Aeris'
plot_CH4timeseries('Utrecht III',pd.to_datetime('2024-06-11 09:48:00'),pd.to_datetime('2024-06-11 17:56:00'), U3_aeris, U3_G2301,save=False, save_path=path_savefig, fig_name='U3_timeseries.pdf',column_names=None)

# Utrecht III -----------------------

fig, ax1 = plt.subplots(figsize=(18,10))
ax2 = ax1.twinx()
ax1.plot(U3_G2301.index,U3_G2301['CH4_ele_G23'], alpha=1, label='G2301', linewidth=2, color=dict_color_instr['G2301'])
ax1.plot(U3_aeris.index,U3_aeris['CH4_ele_aeris'], alpha=1, label='Aeris', linewidth=2, color=dict_color_instr['Aeris'])
ax1.scatter(corrected_peaks_U3.index,corrected_peaks_U3['CH4_ele_G23'], alpha=1, label='Peaks G2301', color='red')
#ax1.scatter(U3_aeris_peaks.index,U3_aeris_peaks['CH4_ele_aeris'], alpha=1, label='Peaks Aeris', color='red')
ax2.scatter(U3_G2301.index,U3_G2301['Speed [m/s]'], alpha=.2, label='Speed m/s', color='orange')
ax3 = ax1.twinx() # Create the third y-axis, sharing the same x-axis
ax3.spines["right"].set_position(("outward", 60))  # Offset the third y-axis to avoid overlap
ax3.plot(U3_G2301.index,U3_G2301['Latitude'], alpha=.8, label='Latitude', color='lightgrey')
ax3.set_ylim(52.085,52.12)
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
ax1.legend()
plt.title('Utrecht III - 2%')



# Rotterdam -----------------------
path_savefig = path_fig + 'Rotterdam/'
starttime       = pd.to_datetime('2022-09-06 06:50:00')
endtime         = pd.to_datetime('2022-09-06 12:59:00')
morning_start   = pd.to_datetime('2022-09-06 07:05:00')
morning_end     = pd.to_datetime('2022-09-06 10:44:00')
afternoon_start = pd.to_datetime("2022-09-06 11:05:00")
afternoon_end = pd.to_datetime('2022-09-06 12:26:00')

lowrr_start = pd.to_datetime('2022-09-06 08:28:00')
lowrr_end = pd.to_datetime('2022-09-06 10:44:00')

R_G2301.name = 'G2301'
R_G4302.name = 'G4302'
R_aeris.name = 'Aeris'
R_miro.name = 'Miro'
R_aerodyne.name = 'Aerodyne'

plot_CH4timeseries('Rotterdam - UU morning',lowrr_start,lowrr_end, R_G4302, R_aeris, R_G2301,save=False, save_path=path_savefig, fig_name='R_timeseries_mUU.pdf',column_names=None)
plot_CH4timeseries('Rotterdam - TNO morning',morning_start,morning_end, R_miro, R_aerodyne,save=False, save_path=path_savefig, fig_name='R_timeseries_mTNO.pdf',column_names=None)
plot_CH4timeseries('Rotterdam - TNO afternoon',afternoon_start,afternoon_end, R_G4302, R_aeris,R_miro, R_aerodyne,save=False, save_path=path_savefig, fig_name='R_timeseries_aTNO.pdf',column_names=None)

fig, ax1 = plt.subplots(figsize=(18,10))
#ax2 = ax1.twinx()
ax1.plot(R_G2301.index,R_G2301['CH4_ele_G23'], alpha=1, label='G2301', linewidth=2, color=dict_color_instr['G2301'])
ax1.plot(R_aeris.index,R_aeris['CH4_ele_aeris'], alpha=1, label='Aeris', linewidth=2, color=dict_color_instr['Aeris'])
# ax1.scatter(corrected_peaks_R_mUU.index,corrected_peaks_R_mUU['CH4_ele_G23'], alpha=1, label='Peaks G2301', color='red')
# ax1.scatter(corrected_peaks_R_mUU.index,corrected_peaks_R_mUU['CH4_ele_aeris'], alpha=1, label='Peaks Aeris', color='red')
#ax1.scatter(R_G2301.index,R_G2301['Speed [m/s]'], alpha=.6, label='Speed m/s', color='orange')
plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
plt.legend()
plt.title('Rotterdam')




#%% Analyse Peaks (Calculate Area)

# from peak_analysis.find_analyze_peaks import *

# analyse_peak is a function in module find_analyze_peaks
# based on the validity of the peaks it  
# 1. assigns 0 or 1 in column QC (0- not valid, 1- valid)
# 2. calculates the integrated area
# 3. assigns the release rate to each peak based on timestamp and location

analyse_peak(corrected_peaks_R_mUU,R_vars_G43,R_vars_aeris, R_vars_G23)
analyse_peak(corrected_peaks_R_mTNO,R_vars_miro,R_vars_aerodyne)
analyse_peak(corrected_peaks_R_aTNO,R_vars_miro,R_vars_aerodyne,R_vars_aeris, R_vars_G43)
 
analyse_peak(corrected_peaks_U,U_vars_G43, U_vars_G23)
analyse_peak_U3(corrected_peaks_U3,U3_vars_G23,U3_vars_aeris)



#%% Treat Data II

# The natural logarithm of the Maximum and Area for each instrument and the Release rate is added to the dataframe

for spec_vars in [R_vars_G43,R_vars_G23,R_vars_aeris]:
    spec = spec_vars['spec']
    corrected_peaks_R_mUU[f'Area_mean_{spec}_log'] = np.log(corrected_peaks_R_mUU[f'Area_mean_{spec}'])
    corrected_peaks_R_mUU[f'Max_{spec}_log']       = np.log(corrected_peaks_R_mUU[f'Max_{spec}'])
corrected_peaks_R_mUU['Release_rate_log']     = np.log(corrected_peaks_R_mUU['Release_rate'])

for spec_vars in [R_vars_miro,R_vars_aerodyne]:
    spec = spec_vars['spec']
    corrected_peaks_R_mTNO[f'Area_mean_{spec}_log'] = np.log(corrected_peaks_R_mTNO[f'Area_mean_{spec}'])
    corrected_peaks_R_mTNO[f'Max_{spec}_log']       = np.log(corrected_peaks_R_mTNO[f'Max_{spec}'])
corrected_peaks_R_mTNO['Release_rate_log']     = np.log(corrected_peaks_R_mTNO['Release_rate'])

for spec_vars in [R_vars_miro,R_vars_aerodyne,R_vars_aeris, R_vars_G43]:
    spec = spec_vars['spec']
    corrected_peaks_R_aTNO[f'Area_mean_{spec}_log'] = np.log(corrected_peaks_R_aTNO[f'Area_mean_{spec}'])
    corrected_peaks_R_aTNO[f'Max_{spec}_log']       = np.log(corrected_peaks_R_aTNO[f'Max_{spec}'])
corrected_peaks_R_aTNO['Release_rate_log']     = np.log(corrected_peaks_R_aTNO['Release_rate'])


for spec_vars in [U_vars_G43,U_vars_G23]:
    spec = spec_vars['spec']
    corrected_peaks_U[f'Area_mean_{spec}_log'] = np.log(corrected_peaks_U[f'Area_mean_{spec}'])
    corrected_peaks_U[f'Max_{spec}_log']       = np.log(corrected_peaks_U[f'Max_{spec}'])
corrected_peaks_U['Release_rate_log']     = np.log(corrected_peaks_U['Release_rate'])

for spec_vars in [U3_vars_G23,U3_vars_aeris]:
    spec = spec_vars['spec']
    corrected_peaks_U3[f'Area_mean_{spec}_log'] = np.log(corrected_peaks_U3[f'Area_mean_{spec}'])
    corrected_peaks_U3[f'Max_{spec}_log']       = np.log(corrected_peaks_U3[f'Max_{spec}'])
corrected_peaks_U3['Release_rate_log']     = np.log(corrected_peaks_U3['Release_rate'])



corrected_peaks_R_mUU['City'] = 'Rotterdam'
corrected_peaks_R_mTNO['City'] = 'Rotterdam'
corrected_peaks_R_aTNO['City'] = 'Rotterdam'
corrected_peaks_U['City'] = 'Utrecht'
corrected_peaks_U3['City'] = 'Utrecht_III'


#%% Save Total Peaks

#from peak_analysis.find_analyze_peaks import *

# Dataframes are filtered for valid peaks (QC = 1) and saved as an excel/csv file

writexlsx = False

def save_to_excel(df,city,name,writexlsx,Day=None):
    total_peaks = df[(df['QC'] == True) & (df['Release_rate'] != 0)].copy(deep=True)
    total_peaks = add_distance_to_df(total_peaks,city,Day=Day) # Add distance to source
    if writexlsx:
        writer = pd.ExcelWriter(path_finaldata+name+".xlsx", engine = 'xlsxwriter')
        total_peaks.to_excel(writer, sheet_name='peaks')
        writer.book.close()
    return total_peaks

def save_to_csv(df,city,name,writexlsx,Day=None):
    total_peaks = df[(df['QC'] == True) & (df['Release_rate'] != 0)].copy(deep=True)
    total_peaks = add_distance_to_df(total_peaks,city,Day=Day) # Add distance to source
    if writexlsx:
        total_peaks.to_csv(path_finaldata+name+".csv")
    
    return total_peaks


# Utrecht
corrected_peaks_U['Loc'] = corrected_peaks_U['Loc'].replace({10: 1, 20: 2})
# total_peaks_U = save_to_excel(corrected_peaks_U,'Utrecht',"U_TOTAL_PEAKS",writexlsx)
total_peaks_U = save_to_csv(corrected_peaks_U,'Utrecht',"U_TOTAL_PEAKS",writexlsx)
total_peaks_U3 = save_to_csv(corrected_peaks_U3,'Utrecht_III',"U3_TOTAL_PEAKS_2perc",writexlsx)
# total_peaks_U3_t5_G2301only = save_to_csv(corrected_peaks_U3_t5_G2301only,'Utrecht_III',"U3_t5_G2301only_TOTAL_PEAKS",writexlsx)
# total_peaks_U3_t5_G2301andAeris = save_to_csv(corrected_peaks_U3_t5_G2301andAeris,'Utrecht_III',"U3_t5_G2301andAeris_TOTAL_PEAKS",writexlsx)

# Rotterdam
peaks_UUm      = corrected_peaks_R_mUU[corrected_peaks_R_mUU['QC']]
peaks_TNOm       = corrected_peaks_R_mTNO[corrected_peaks_R_mTNO['QC']]
peaks_TNOa = corrected_peaks_R_aTNO[corrected_peaks_R_aTNO['QC']]
cars            = ['UUAQ', 'TNO']

total_peaks_R     = pd.concat((peaks_UUm, peaks_TNOm, peaks_TNOa))
total_peaks_R['Loc'] = total_peaks_R['Loc'].replace({10: 1, 20: 2})
total_peaks_R.index.name = 'Datetime'
add_distance_to_df(total_peaks_R,'Rotterdam')


#writexlsx=True
if writexlsx:
    total_peaks_R.to_csv(path_finaldata+"R_TOTAL_PEAKS.csv")
    # path_save      = path_finaldata+"R_TOTAL_PEAKS.xlsx"
    # writer         = pd.ExcelWriter(path_save, engine = 'xlsxwriter')
    # total_peaks_R.to_excel(writer, sheet_name='peaks')
    # writer.book.close()


# RU2 ------

total_peaks_all     = pd.concat((total_peaks_R, total_peaks_U,total_peaks_U3))

writexlsx = False
if writexlsx:
    total_peaks_all.to_csv(path_finaldata+'RU2_TOTAL_PEAKS.csv')

    

# #%% P: Detailed Peak Plots


# ''' ===== Utrecht III ===== '''

# # Define other necessary variables
# coord_extent_1 = [total_peaks_U3['Longitude'].min()-0.001,total_peaks_U3['Longitude'].max(), total_peaks_U3['Latitude'].min()-0.0005, total_peaks_U3['Latitude'].max()+0.0005] #r_loc: 43.782970N, (-)79.46952W 
# coord_extent_1 = [5.1633, 5.166, 52.0873, 52.0888]
# release_loc1 = (5.164652777777778, 52.0874472)
# release_loc1_2 = (5.16506388888889, 52.0875333) 
# release_loc2 = (5.164452777777778, 52.0885333) # 
# # column_names_1 = {'G2301': 'CH4_ele_G23'}
# # column_names_2 = {'Aeris': 'CH4_ele_aeris'}
# column_names_1 = {'G2301': 'CH4_ele_G23','Aeris': 'CH4_ele_aeris'}
# indiv_peak_plots = True  # or False based on your requirement


# path_savefig = path_fig + "Utrecht_III/Peakplots_QCpassed"


# # First location on lane 1 (in excel file named location 3)
# plot_indivpeaks_afterQC(total_peaks_U3[:'2024-06-11 11:22:00'], path_fig, coord_extent_1, release_loc1, release_loc2, indiv_peak_plots, column_names_1, U3_vars_G23, U3_vars_aeris)

# # Second location on lane 1 (in excel file named location 1)
# plot_indivpeaks_afterQC(total_peaks_U3['2024-06-11 11:22:00':], path_fig, coord_extent_1, release_loc1_2, release_loc2, indiv_peak_plots, column_names_1, U3_vars_G23, U3_vars_aeris)


# ''' ===== Rotterdam ===== '''


# # Define other necessary variables
# coord_extent = [4.51832, 4.52830, 51.91921, 51.92288]
# release_loc1_R = (4.5237450, 51.9201216)
# release_loc2_R = (4.5224917, 51.9203931) #51.9203931,4.5224917
# release_loc3_R = (4.523775, 51.921028) # estimated from Daans plot (using google earth)
# column_names_mUU = {'G4302': 'CH4_ele_G43','G2301': 'CH4_ele_G23', 'Aeris':'CH4_ele_aeris'}
# column_names_mTNO = {'Miro': 'CH4_ele_miro', 'Aerodyne': 'CH4_ele_aero'}
# column_names_aTNO = {'Miro': 'CH4_ele_miro', 'Aerodyne': 'CH4_ele_aero', 'G4302': 'CH4_ele_G43', 'Aeris':'CH4_ele_aeris'}
# indiv_peak_plots = True  # or False based on your requirement


# ''' --- Morning UU --- '''

# path_savefig =  path_fig + "Rotterdam/Peakplots_QCpassed/Morning_UU"
# # Call the function with necessary arguments
# plot_indivpeaks_afterQC(corrected_peaks_R_mUU, path_savefig, coord_extent, release_loc1_R, release_loc2_R, indiv_peak_plots, column_names_mUU, R_vars_G43, R_vars_G23,R_vars_aeris)
# # Note: put main instrument first in args* (for Morning UU: G4302 -> R_vars_G43)

# ''' --- Morning TNO --- '''

# path_savefig =  path_fig + "Rotterdam/Peakplots_QCpassed/Morning_TNO"
# # Call the function with necessary arguments
# plot_indivpeaks_afterQC(corrected_peaks_R_mTNO, path_savefig, coord_extent, release_loc1_R, release_loc2_R, indiv_peak_plots, column_names_mTNO, R_vars_miro, R_vars_aerodyne)


# ''' --- Afternoon TNO --- '''

# path_savefig =  path_fig + "Rotterdam/Peakplots_QCpassed/Afternoon_TNO"
# # Call the function with necessary arguments
# plot_indivpeaks_afterQC(corrected_peaks_R_aTNO, path_savefig, coord_extent, release_loc1_R, release_loc3_R, indiv_peak_plots, column_names_aTNO, R_vars_miro, R_vars_aerodyne,R_vars_G43,R_vars_aeris)





#%% Process&Save for further analysis

#from peak_analysis.find_analyze_peaks import *
# Combine -------------------------------------------------------------------------------------------------------------------------------

df_U1 = total_peaks_U[['Loc','Release_rate','Area_mean_G23','Area_mean_G43','Max_G23','Max_G43','Latitude','Longitude','Mean_speed']].copy(deep=True)
df_U3 = total_peaks_U3[['Loc','Release_rate','Area_mean_G23','Area_mean_aeris','Max_G23','Max_aeris','Latitude','Longitude','Mean_speed']].copy(deep=True)

# Rotterdam ------------------
df_R_comb = total_peaks_R.copy(deep=True)
max_columns = df_R_comb.filter(like='Max').copy(deep=True)
df_R_comb.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Loc'] = df_R_comb['Loc']
max_columns['Release_rate'] = df_R_comb['Release_rate']
max_columns['Loc_tuple'] = max_columns.apply(combine_columns, axis=1)
max_columns.drop(['Loc','Release_rate'],axis=1,inplace=True)
df_R_comb.reset_index(inplace=True,drop=False)
df_R_comb = pd.melt(df_R_comb, id_vars=['Loc','Release_rate','Longitude', 'Latitude','Datetime','Peak','Mean_speed'],
                    value_vars=['Area_mean_aeris', 'Area_mean_G23', 'Area_mean_G43','Area_mean_miro','Area_mean_aero'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Loc_tuple'], value_vars=['Max_aeris', 'Max_G23', 'Max_G43', 'Max_miro', 'Max_aero'], 
                      var_name='Instruments_max', value_name='Max')
df_R_comb = pd.concat([df_R_comb, max_columns], axis=1)
df_R_comb.dropna(subset=['Area'], inplace=True)
df_R_comb.drop(['Instruments_area'],axis=1,inplace=True)

# Claculate log
df_R_comb['ln(Area)'] = np.log(df_R_comb['Area'])
df_R_comb['ln(Max)'] = np.log(df_R_comb['Max'])

df_R_rr_count = df_R_comb.groupby(['Release_rate']).size().reset_index(name='count')

# Rotterdam UU and TNO separate -----------------------
# first extract right instruments
R1 = total_peaks_R[:'2022-09-06 11:05:00'].copy(deep=True)
R2 = total_peaks_R['2022-09-06 11:05:00':].copy(deep=True)
df_R_comb_UU = R1[R1['Max_G23'].notna()].copy(deep=True) #only in the morning
df_R_comb_TNO = R1[R1['Max_miro'].notna()].copy(deep=True)
df_R_comb_TNO = pd.concat([df_R_comb_TNO,R2]) # in the morning + afternoon

# TNO
max_columns = df_R_comb_TNO.filter(like='Max').copy(deep=True)
df_R_comb_TNO.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Loc'] = df_R_comb_TNO['Loc']
max_columns['Release_rate'] = df_R_comb_TNO['Release_rate']
max_columns['Loc_tuple'] = max_columns.apply(combine_columns, axis=1)
max_columns.drop(['Loc','Release_rate'],axis=1,inplace=True)
df_R_comb_TNO = pd.melt(df_R_comb_TNO, id_vars=['Loc','Release_rate','Longitude', 'Latitude','Mean_speed'],
                    value_vars=['Area_mean_G43', 'Area_mean_aeris','Area_mean_miro','Area_mean_aero'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Loc_tuple'], value_vars=['Max_G43','Max_aeris', 'Max_miro', 'Max_aero'], 
                      var_name='Instruments_max', value_name='Max')
df_R_comb_TNO = pd.concat([df_R_comb_TNO, max_columns], axis=1)
df_R_comb_TNO.dropna(subset=['Area'], inplace=True)
df_R_comb_TNO.drop(['Instruments_area'],axis=1,inplace=True)

df_R_comb_TNO['ln(Area)'] = np.log(df_R_comb_TNO['Area']) # Claculate log
df_R_comb_TNO['ln(Max)'] = np.log(df_R_comb_TNO['Max'])

df_R_TNO_rr_count = df_R_comb_TNO.groupby(['Release_rate']).size().reset_index(name='count')

# UU
max_columns = df_R_comb_UU.filter(like='Max').copy(deep=True)
df_R_comb_UU.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Loc'] = df_R_comb_UU['Loc']
max_columns['Release_rate'] = df_R_comb_UU['Release_rate']
max_columns['Loc_tuple'] = max_columns.apply(combine_columns, axis=1)
max_columns.drop(['Loc','Release_rate'],axis=1,inplace=True)
df_R_comb_UU = pd.melt(df_R_comb_UU, id_vars=['Loc','Release_rate','Longitude', 'Latitude','Mean_speed'],
                    value_vars=['Area_mean_aeris', 'Area_mean_G23', 'Area_mean_G43'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Loc_tuple'], value_vars=['Max_aeris', 'Max_G23', 'Max_G43'], 
                      var_name='Instruments_max', value_name='Max')
df_R_comb_UU = pd.concat([df_R_comb_UU, max_columns], axis=1)
df_R_comb_UU.dropna(subset=['Area'], inplace=True)
df_R_comb_UU.drop(['Instruments_area'],axis=1,inplace=True)

df_R_comb_UU['ln(Area)'] = np.log(df_R_comb_UU['Area']) # Claculate log
df_R_comb_UU['ln(Max)'] = np.log(df_R_comb_UU['Max'])

df_R_UU_rr_count = df_R_comb_UU.groupby(['Release_rate']).size().reset_index(name='count')



# Utrecht ------------------
df_U1_comb = df_U1.copy(deep=True)
max_columns = df_U1_comb.filter(like='Max').copy(deep=True)
df_U1_comb.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Loc'] = df_U1_comb['Loc']
max_columns['Release_rate'] = df_U1_comb['Release_rate']
max_columns['Loc_tuple'] = max_columns.apply(combine_columns, axis=1)
max_columns.drop(['Loc','Release_rate'],axis=1,inplace=True)
df_U1_comb.reset_index(inplace=True,drop=False)
df_U1_comb = pd.melt(df_U1_comb, id_vars=['Loc','Release_rate','Latitude','Longitude','Mean_speed','Datetime'],
                    value_vars=['Area_mean_G23', 'Area_mean_G43'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Loc_tuple'], value_vars=['Max_G23', 'Max_G43'], 
                      var_name='Instruments_max', value_name='Max')
df_U1_comb = pd.concat([df_U1_comb, max_columns], axis=1)
df_U1_comb.drop(['Instruments_area'],axis=1,inplace=True)

# Calculate log
df_U1_comb['ln(Area)'] = np.log(df_U1_comb['Area'])
df_U1_comb['ln(Max)'] = np.log(df_U1_comb['Max'])

df_U1_rr_count = df_U1_comb.groupby(['Release_rate']).size().reset_index(name='count')

# Utrecht III ------------------
df_U3_comb = df_U3.copy(deep=True)
max_columns = df_U3_comb.filter(like='Max').copy(deep=True)
df_U3_comb.drop(list(max_columns.columns),axis=1,inplace=True) #must be directly after the filter line, sicne later additional columns are added which should not be droped from df_R
max_columns['Loc'] = df_U3_comb['Loc']
max_columns['Release_rate'] = df_U3_comb['Release_rate']
max_columns['Loc_tuple'] = max_columns.apply(combine_columns, axis=1)
max_columns.drop(['Loc','Release_rate'],axis=1,inplace=True)
df_U3_comb.reset_index(inplace=True,drop=False)
df_U3_comb = pd.melt(df_U3_comb, id_vars=['Loc','Release_rate','Latitude','Longitude','Mean_speed','Datetime'],
                    value_vars=['Area_mean_G23', 'Area_mean_aeris'], 
                    var_name='Instruments_area', value_name='Area')
max_columns = pd.melt(max_columns, id_vars=['Loc_tuple'], value_vars=['Max_G23', 'Max_aeris'], 
                      var_name='Instruments_max', value_name='Max')
df_U3_comb = pd.concat([df_U3_comb, max_columns], axis=1)
df_U3_comb.drop(['Instruments_area'],axis=1,inplace=True)

# Calculate log
df_U3_comb['ln(Area)'] = np.log(df_U3_comb['Area'])
df_U3_comb['ln(Max)'] = np.log(df_U3_comb['Max'])

df_U3_rr_count = df_U3_comb.groupby(['Release_rate']).size().reset_index(name='count')





# Add distance to source ------------------
add_distance_to_df(df_R_comb,'Rotterdam')
add_distance_to_df(df_U1_comb,'Utrecht')
add_distance_to_df(df_U3_comb,'Utrecht_III')


df_R_comb['City'] = 'Rotterdam'
df_U1_comb['City'] = 'Utrecht'
df_U3_comb['City'] = 'Utrecht_III'


# Merge all cities into one df------------------
# reset index to have it also in df_all

R = df_R_comb.copy()
R.reset_index(inplace=True,drop=False)
U1 = df_U1_comb.copy()
U1.reset_index(inplace=True,drop=False)
U3 = df_U3_comb.copy()
U3 = U3.dropna(subset=['Max']) # drop rows where Max is NaN
U3.reset_index(inplace=True,drop=False)



df_RU2 = [R,U1,U3]
# Concatenate the DataFrames vertically
df_RU2 = pd.concat(df_RU2, axis=0, ignore_index=True, sort=False)
df_RU2 = df_RU2.set_index(df_RU2.columns[0], drop=True) # set Datetime column as index
df_RU2_rr_count = df_RU2.groupby(['Release_rate']).size().reset_index(name='count')

save_to_csv=True

if save_to_csv:
    # Save the DataFrame as a CSV file
    #
    df_RU2.to_csv(path_finaldata+'RU2_TOTAL_PEAKS_comb.csv')
    df_RU2_rr_count.to_csv(path_finaldata+'RU2_count.csv')
    # release height =4 in London NOT included
    
    
    df_R_comb.to_csv(path_finaldata+'R_comb_final.csv')
    # df_R_comb_TNO.to_csv(path_finaldata+'R_TNO_comb_final.csv')
    # df_R_comb_UU.to_csv(path_finaldata+'R_UU_comb_final.csv')
    df_U1_comb.to_csv(path_finaldata+'U1_comb_final.csv')
    df_U3_comb.to_csv(path_finaldata+'U3_comb_final.csv')
    df_Ld2_comb.to_csv(path_finaldata+'Ld2_comb_final.csv')
    
    df_R_rr_count.to_csv(path_finaldata+'R_comb_count.csv')
    # df_R_TNO_rr_count.to_csv(path_finaldata+'R_TNO_comb_count.csv')
    # df_R_UU_rr_count.to_csv(path_finaldata+'R_UU_comb_count.csv')
    df_U1_rr_count.to_csv(path_finaldata+'U1_comb_count.csv')
    df_U3_rr_count.to_csv(path_finaldata+'U3_comb_count.csv')
    print('Dataframes saved as .csv files')

print('Script finished')


#%% End
