# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:32:30 2024

@author: Judith

Collection of functions necessary for finding and analyzing peaks.
To be used in file ? postprocessing_script.py ?

"""

# Import necessary packages
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from geopy.distance import geodesic


from plotting.general_plots import *



def find_right_side_index(df,row,CH4_column):
    left_base_index = row['Peakstart'] #left base index
    peak_max_index = row.name # peak max
    print(peak_max_index)
    
    left_base_value = df.loc[left_base_index, CH4_column]
    portion_after_peak = df.loc[peak_max_index:peak_max_index+ pd.Timedelta(minutes=1)].copy()
   
    # Find the first index where the value is within the range of left base value plus/minus 2
    tolerance = 0.01
    matching_indices = portion_after_peak.index[(portion_after_peak[CH4_column] >= left_base_value - tolerance) & (portion_after_peak[CH4_column] <= left_base_value + tolerance)]
    
    if len(matching_indices) > 0:
        # Get the first index from the matching indices
        index_of_closest_value = matching_indices[0]
    else:
        # If no matching index found, handle the case as per your requirement
        # For example, raise an exception or set index_of_closest_value to a default value.
        index_of_closest_value = portion_after_peak[CH4_column].sub(left_base_value).abs().idxmin()
    return index_of_closest_value




def improve_base_indices(df,row,CH4_column):
    left_base_index = row['Peakstart'] #left base index
    left_base_index_new = left_base_index
    peak_max_index = row.name # peak max
    print(peak_max_index)
    #---new 04.04.24
    # peak_length = (row['Peakend'] - row['Peakstart']).dt.total_seconds()
    # peak_length_1stpart = (row.index - row['Peakstart']).dt.total_seconds()
    # peak_length_2ndpart = (row['Peakend'] - row.index).dt.total_seconds()
    
    left_base_value = df.loc[left_base_index, CH4_column]
    portion_after_peak = df.loc[peak_max_index:peak_max_index+ pd.Timedelta(minutes=1)].copy()
   
    # Find the first index where the value is within the range of left base value plus/minus 2
    tolerance = 0.01
    matching_indices = portion_after_peak.index[(portion_after_peak[CH4_column] >= left_base_value - tolerance) & (portion_after_peak[CH4_column] <= left_base_value + tolerance)]
    
    if len(matching_indices) > 0:
        # Get the first index from the matching indices
        right_base_index_new = matching_indices[0]
    else:
        # If no matching index found, handle the case as per your requirement
        # For example, raise an exception or set index_of_closest_value to a default value.
        right_base_index_new = portion_after_peak[CH4_column].sub(left_base_value).abs().idxmin()
    
    #---new 04.04.24
    peak_length = (right_base_index_new - row['Peakstart']).total_seconds()
    peak_length_1stpart = (row.name - row['Peakstart']).total_seconds()
    peak_length_2ndpart = (right_base_index_new - row.name).total_seconds()
    
    if peak_length > 120: #(mean length is U:18s - T1b:69s)
        if peak_length_1stpart > peak_length_2ndpart:
            left_base_index_new = peak_max_index - timedelta(seconds=20)
        else:
            right_base_index_new = peak_max_index + timedelta(seconds=60) 
    #---
    return left_base_index_new, right_base_index_new



def process_peak_data_new(df, spec, distance=3, height=None, width=None, overviewplot=False, savepath = None):
    # changed 4.4.24
    # 5.8.24 deleted writer&writxlsx in function, was not used anyway
    df = df.copy() #.loc[morning_start:morning_end]
    bg = df[('bg_'+spec)] #spec = G2 or aero
    CH4data = df[('CH4_'+spec)]
    
    if not height:
        height = 1.1

    scp_peaks, properties = find_peaks(CH4data.values, height=height * bg.values, distance=distance, width=width)
    # TEST TEST TEST TEST TEST TEST TTEST
    #scp_peaks, properties = find_peaks(CH4data.values, height=None, distance=distance, width=width)
    
    N = len(scp_peaks)
    print(f'Found {N} peaks for {spec}')

    peakdf = df.iloc[scp_peaks].copy()
    peakdf['peak max'] = np.around(properties['peak_heights'], 2)
    peakdf['Peakstart'] = df.iloc[properties['left_bases']].index
    #right_side_indices = peakdf.apply(find_right_side_index, axis=1)
    base_indices = peakdf.apply(lambda row: improve_base_indices(df, row, ('CH4_'+spec)), axis=1)
    print(base_indices.info())
    print(base_indices)
    
    # Assuming base_indices is a Pandas Series containing tuples
    base_indices_df = base_indices.apply(pd.Series)
    
    # Rename the columns
    base_indices_df.columns = ['Left_Base_Index', 'Right_Base_Index']

    peakdf['Peakstart'] = base_indices_df.iloc[:,0]
    peakdf['Peakend'] = base_indices_df.iloc[:,1]
    peakdf['Width (s)'] = (peakdf['Peakend'] - peakdf['Peakstart']).dt.seconds
    #peakdf['BG'] = bg[scp_peaks].copy()
    peakdf = peakdf.rename(columns={'bg_'+spec: 'BG'})

    peakdf.index = pd.to_datetime(peakdf.index)
    peakdf.index = peakdf.index.strftime('%Y-%m-%d %H:%M:%S.%f')

    # savepath = None
    # if path_fig:
    #     savepath = path_fig + f"U_Peakplots/peakfinder_{spec}.jpg"

    if overviewplot:
        overview_plot(CH4data, scp_peaks, spec, N, bg=bg, th=height * bg, savepath=savepath)
    
    return peakdf


def process_peak_data(df, spec, distance=3, width=None, writexlsx=False, writer=None, overviewplot=False, savepath = None):
    df = df.copy() #.loc[morning_start:morning_end]
    bg = df[('bg_'+spec)] #spec = G2 or aero
    CH4data = df[('CH4_'+spec)]

    scp_peaks, properties = find_peaks(CH4data.values, height=1.1 * bg.values, distance=distance, width=width)
    N = len(scp_peaks)
    print(f'Found {N} peaks for {spec}')

    peakdf = df.iloc[scp_peaks].copy()
    peakdf['peak max'] = np.around(properties['peak_heights'], 2)
    peakdf['Peakstart'] = df.iloc[properties['left_bases']].index
    #right_side_indices = peakdf.apply(find_right_side_index, axis=1)
    right_side_indices = peakdf.apply(lambda row: find_right_side_index(df, row, ('CH4_'+spec)), axis=1)

    peakdf['Peakend'] = right_side_indices
    peakdf['Width (s)'] = (peakdf['Peakend'] - peakdf['Peakstart']).dt.seconds
    #peakdf['BG'] = bg[scp_peaks].copy()
    peakdf = peakdf.rename(columns={'bg_'+spec: 'BG'})

    peakdf.index = pd.to_datetime(peakdf.index)
    peakdf.index = peakdf.index.strftime('%Y-%m-%d %H:%M:%S.%f')

    # savepath = None
    # if path_fig:
    #     savepath = path_fig + f"U_Peakplots/peakfinder_{spec}.jpg"

    if overviewplot:
        overview_plot(CH4data, scp_peaks, spec, N, bg=bg, th=1.1 * bg, savepath=savepath)
    
    return peakdf


def releaserate_U(peaktime, loca,day):
        
        # Location 1
        L1_RR1          = pd.to_datetime('2022-11-25 12:06:40')
        T_lunch         = pd.to_datetime('2022-11-25 12:46:00')
        L1_RR2          = pd.to_datetime('2022-11-25 13:22:00')
        end             = pd.to_datetime('2022-11-25 14:17:00')
        
        # Location 2
        L2_RR1          = pd.to_datetime('2022-11-25 12:06:40')
        L2_RR2          = pd.to_datetime('2022-11-25 13:22:00')    
        
        peaktime = pd.to_datetime(peaktime)
        
        if (loca == 1.0) or (loca == 10.0):    
            if peaktime >= L1_RR1:
                if peaktime < T_lunch:
                    release_rate = 3
                    RR = '3 L/h'
                elif peaktime < L1_RR2:
                    release_rate = None
                    RR = None
                elif peaktime < end:
                    release_rate = 2.18 
                    RR = '2.18 L/h'
                else:
                    print('weird mistake')
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/m'
        elif (loca == 2.0) or (loca == 20.0):
            if peaktime >= L2_RR1:
                if peaktime < T_lunch:
                    release_rate = 15
                    RR = '15 L/h'
                elif peaktime < L1_RR2:
                    release_rate = None
                    RR = None
                elif peaktime < end:
                    release_rate = 15 
                    RR = '15 L/h'
                else:
                    print('weird mistake')
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/m'
        return RR, release_rate


def releaserate_U3(peaktime, loca, day):
        
        # Location 1
        loc1_t_4Lm      = pd.to_datetime('2024-06-11 10:48:00')
        loc1_t_0Lm      = pd.to_datetime('2024-06-11 11:22:00')
        loc1_t2_4Lm     = pd.to_datetime('2024-06-11 11:38:00')
        loc1_t_10Lm     = pd.to_datetime('2024-06-11 11:48:00')
        loc1_t_80Lm     = pd.to_datetime('2024-06-11 12:18:00')
        loc1_t_20Lm     = pd.to_datetime('2024-06-11 12:32:00')
        t_lunch         = pd.to_datetime('2024-06-11 13:01:00')
        loc1_t_100Lm    = pd.to_datetime('2024-06-11 14:05:00')
        loc1_t_40Lm     = pd.to_datetime('2024-06-11 14:21:00')
        loc1_t3_0Lm     = pd.to_datetime('2024-06-11 14:23:00')
        loc1_t_15Lm     = pd.to_datetime('2024-06-11 14:28:00')
        loc1_t4_0Lm     = pd.to_datetime('2024-06-11 15:23:00')
        loc1_t3_4Lm      = pd.to_datetime('2024-06-11 16:19:00')
        loc1_t_015Lm    = pd.to_datetime('2024-06-11 17:07:00')
        loc1_t_1Lm      = pd.to_datetime('2024-06-11 17:44:00')
        loc1_t_end      = pd.to_datetime('2024-06-11 17:56:00')
        
        # Location 2
        # loc2_t_2_4Lm    = pd.to_datetime('2024-06-11 10:50:00')
        # loc2_t_2_5Lm    = pd.to_datetime('2024-06-11 11:10:00')
        # loc2_t_2_6Lm    = pd.to_datetime('2024-06-11 11:29:00')
        loc2_t_2_5Lm    = pd.to_datetime('2024-06-11 10:50:00')
        loc2_t_4Lm      = pd.to_datetime('2024-06-11 11:31:00')
        loc2_t_05Lm     = pd.to_datetime('2024-06-11 12:09:00')
        loc2_t_015Lm    = pd.to_datetime('2024-06-11 12:37:00')
        t_lunch         = pd.to_datetime('2024-06-11 13:01:00')
        loc2_t_03Lm     = pd.to_datetime('2024-06-11 14:02:00')
        loc2_t_2Lm      = pd.to_datetime('2024-06-11 14:40:00')
        loc2_t_1Lm      = pd.to_datetime('2024-06-11 15:13:00')
        loc2_t_0Lm      = pd.to_datetime('2024-06-11 15:52:00')
        loc2_t2_4Lm     = pd.to_datetime('2024-06-11 17:00:00')
        loc2_t_60Lm     = pd.to_datetime('2024-06-11 17:30:00')
        loc2_t_20Lm     = pd.to_datetime('2024-06-11 17:38:00')
        loc2_t_80Lm     = pd.to_datetime('2024-06-11 18:06:00')
        loc2_t_end      = pd.to_datetime('2024-06-11 18:25:00')
        
        #print(loca)
        
        if ((loca == 1.0) | (loca == 10)| (loca == 3) | (loca == 30)): 
            if peaktime >= loc1_t_4Lm:
                if peaktime < loc1_t_0Lm:
                    release_rate = 4
                    RR = '4 L/min'
                elif peaktime < loc1_t2_4Lm:
                    release_rate = 0
                    RR = '0 L/min'
                elif peaktime < loc1_t_10Lm:
                    release_rate = 3.95 
                    RR = '3.95 L/min'
                elif peaktime < loc1_t_80Lm:
                    release_rate = 10 
                    RR = '10 L/min'
                elif peaktime < loc1_t_20Lm:
                    release_rate = 80 
                    RR = '80 L/min'
                elif peaktime < t_lunch:
                    release_rate = 20 
                    RR = '20 L/min'
                elif peaktime < loc1_t_100Lm:
                    release_rate = 0
                    RR = '0 L/min'
                elif peaktime < loc1_t_40Lm:
                    release_rate = 100
                    RR = '100 L/min'
                elif peaktime < loc1_t3_0Lm:
                    release_rate = 0
                    RR = '0 L/min'
                elif peaktime  < loc1_t_15Lm:
                    release_rate = 0
                    RR = '0 L/min'
                elif peaktime  < loc1_t4_0Lm:
                    release_rate = 15
                    RR = '15 L/min'
                elif peaktime  < loc1_t3_4Lm:
                    release_rate = 0
                    RR = '0 L/min'
                elif peaktime  < loc1_t_015Lm:
                    release_rate = 4
                    RR = '4 L/min'
                elif peaktime  < loc1_t_1Lm:
                    release_rate = 0.15
                    RR = '0.15 L/min'
                elif peaktime  < loc1_t_end:
                    release_rate = 1
                    RR = '1 L/min'
                else:
                    print(f'{peaktime} not in release period (after end of release)')
                    release_rate = 0
                    RR = '0 L/min'
            else: 
                print(f'{peaktime} not in release period (before start of release)')
                release_rate = 0
                RR = '0 L/min'
                
        elif ((loca == 2.0) | (loca == 20)):
            if peaktime >= loc2_t_2_5Lm:
                if peaktime < loc2_t_4Lm:
                    release_rate = 2.5
                    RR = '2.5 L/min'
                elif peaktime < loc2_t_05Lm:
                    release_rate = 4
                    RR = '4 L/min'
                elif peaktime < loc2_t_015Lm:
                    release_rate = 0.5
                    RR = '0.5 L/min'
                elif peaktime < t_lunch:
                    release_rate = 0.15
                    RR = '0.15 L/min'
                elif peaktime < loc2_t_03Lm:
                    release_rate = 0
                    RR = '0 L/min'
                elif peaktime < loc2_t_2Lm:
                    release_rate = 0.3
                    RR = '0.3 L/min'
                elif peaktime < loc2_t_1Lm:
                    release_rate = 2.2
                    RR = '2.2 L/min'
                elif peaktime < loc2_t_0Lm:
                    release_rate = 1
                    RR = '1 L/min'
                elif peaktime < loc2_t2_4Lm:
                    release_rate = 0
                    RR = '0 L/min'
                elif peaktime < loc2_t_60Lm:
                    release_rate = 4
                    RR = '4 L/min'
                elif peaktime < loc2_t_20Lm:
                    release_rate = 60
                    RR = '60 L/min'
                elif peaktime < loc2_t_80Lm:
                    release_rate = 20
                    RR = '20 L/min'
                elif peaktime < loc2_t_end:
                    release_rate = 80
                    RR = '80 L/min'
                else:
                    print(f'{peaktime} not in release period (after end of release)')
                    release_rate = 0
                    RR = '0 L/min'
            else: 
                print(f'{peaktime} not in release period (before start of release)')
                release_rate = 0
                RR = '0 L/min'
                
        else:
           print(f'{peaktime}: {loca} not a valid location') 
           RR='0 L/min'
           release_rate = 0
                
        return RR, release_rate


def releaserate_R(peaktime, loca, day):
        
        # Location 1
        T_5Lm           = pd.to_datetime('2022-09-06 07:05:00')
        T_10Lm          = pd.to_datetime('2022-09-06 08:15:00')
        T_20Lm          = pd.to_datetime('2022-09-06 08:58:00')
        T_40Lm          = pd.to_datetime('2022-09-06 09:23:20')
        T_80Lm          = pd.to_datetime('2022-09-06 09:54:00')
        T_lunch         = pd.to_datetime('2022-09-06 10:44:00')
        T2_20Lm         = pd.to_datetime('2022-09-06 11:05:00')
        T_120Lm         = pd.to_datetime('2022-09-06 11:34:00')
        T2_40Lm         = pd.to_datetime('2022-09-06 11:48:00')
        end             = pd.to_datetime('2022-09-06 12:26:00')
        
        # Location 2
        loc2_T_1L       = pd.to_datetime('2022-09-06 08:28:00')
        loc2_T_150mL    = pd.to_datetime('2022-09-06 09:01:00')
        loc2_T_515mL    = pd.to_datetime('2022-09-06 09:39:00')
        loc2_T_310mL    = pd.to_datetime('2022-09-06 10:12:00')
        
        # Location 3
        Loc3_T_3Lm     = pd.to_datetime('2022-09-06 11:16:00')
        
        
        if loca == 1.0:    
            if peaktime >= T_5Lm:
                if peaktime < T_10Lm:
                    release_rate = 5
                    RR = '5 L/min'
                elif peaktime < T_20Lm:
                    release_rate = 10
                    RR = '10 L/min'
                elif peaktime < T_40Lm:
                    release_rate = 20 
                    RR = '20 L/min'
                elif peaktime < T_80Lm:
                    release_rate = 40 
                    RR = '40 L/min'
                elif peaktime < T_lunch:
                    release_rate = 80 
                    RR = '80 L/min'
                elif peaktime < T2_20Lm:
                    release_rate = None
                    RR = None
                elif peaktime < T_120Lm:
                    release_rate = 20
                    RR = '20 L/min'
                elif peaktime < T2_40Lm:
                    release_rate = 120
                    RR = '120 L/min'
                elif peaktime  < end:
                    release_rate = 40
                    RR = '40 L/min'
                else:
                    print('weird mistake')
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/min'
        elif loca == 2.0:
            if peaktime >= loc2_T_1L:
                if peaktime < loc2_T_150mL:
                    release_rate = 1
                    RR = '1 L/min'
                elif peaktime < loc2_T_515mL:
                    release_rate = 0.150
                    RR = '150 mL/min'
                elif peaktime < loc2_T_310mL:
                    release_rate = 0.515
                    RR = '515 mL/min'
                elif peaktime < end: # should be T_lunch not end
                    release_rate = 0.310
                    RR = '310 mL/min'
                else:
                    print('weird mistake')
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/min'
        elif loca == 3.0:
            if peaktime > Loc3_T_3Lm:
                if peaktime < end:
                    release_rate = 3.33 # chenged 25.9 (before 200/60)
                    RR = '3.33 L/min'
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/min'
                
        else:
           print(peaktime, loca) 
           RR='0 L/min'
           release_rate = 0
                
        return RR, release_rate
    
def releaserate_T(peaktime,loca,day):
        # Toronto    
    
        rho_CH4 = 0.662 #kg/m^3 or g/L 
 
        if (day=='Day1-bike') or (day=='Day1-car'):
            # r1_start = datetime(2021,10,20,20,13)
            # r1_finish = datetime(2021,10,20,20,18)
            # r2_start = datetime(2021,10,20,20,19)
            # r2_finish = datetime(2021,10,20,20,25)
            # r3_start = datetime(2021,10,20,20,30)
            # r3_finish = datetime(2021,10,20,20,40)
            # r4_start = datetime(2021,10,20,20,40)
            # r4_finish = datetime(2021,10,20,20,49)   
            
            r1_start = datetime(2021,10,20,20,11)
            r1_finish = datetime(2021,10,20,20,18)
            r2_start = datetime(2021,10,20,20,19)
            r2_finish = datetime(2021,10,20,20,27) #I assume 27, check with Lawson
            r3_start = datetime(2021,10,20,20,30)
            r3_finish = datetime(2021,10,20,20,40)
            r4_start = datetime(2021,10,20,20,40)
            r4_finish = datetime(2021,10,20,20,49) 
                    
            peaktime = pd.to_datetime(peaktime)
            
            if peaktime >= r1_start:
                if peaktime < r2_start:
                    # 393.6 g/h
                    release_rate = round(393.6/rho_CH4/60,1) # L/min
                    RR = f'{release_rate} L/min'
                elif peaktime < r3_start:
                    # 196.8 g/h
                    release_rate = round(196.8/rho_CH4/60,1) # L/min
                    RR = f'{release_rate} L/min'
                elif peaktime < r4_start:
                    # 98.4 g/h
                    release_rate = round(98.4/rho_CH4/60,1) # L/min
                    RR = f'{release_rate} L/min'
                elif peaktime < r4_finish:
                    # 787.2 g/h ?
                    release_rate = round(787.2/rho_CH4/60,1) # L/min
                    RR = f'{release_rate} L/min'
                else:
                    print('weird mistake')
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/min'
                
        elif (day=='Day2-car'):
            r1_start = datetime(2021,10,24,13,48)
            r1_finish = datetime(2021,10,24,13,58)
            r2_start = datetime(2021,10,24,14,3)
            r2_finish = datetime(2021,10,24,14,11)
            r3_start = datetime(2021,10,24,14,16)
            r3_finish = datetime(2021,10,24,14,24)
            r4_start = datetime(2021,10,24,14,28)
            r4_finish = datetime(2021,10,24,14,37)
            r5_start = datetime(2021,10,24,14,41)
            r5_finish = datetime(2021,10,24,14,59)
                    
            peaktime = pd.to_datetime(peaktime)
            
            if peaktime >= r1_start:
                if peaktime < r2_start:
                    # 393.6 g/h - 9.9 L/min
                    release_rate = round(393.6/rho_CH4/60,1) # L/min
                    RR = f'{release_rate} L/min'
                elif peaktime < r3_start:
                    # 196.8 g/h - 5 L/min
                    release_rate = round(196.8/rho_CH4/60,1) # L/min
                    RR = f'{release_rate} L/min'
                elif peaktime < r4_start:
                    # 39.36 g/h = 1 L/min
                    release_rate = round(39.36/rho_CH4/60,1) # L/min
                    RR = f'{release_rate} L/min'
                elif peaktime < r4_finish:
                    # 4.92 g/h = 0.12 L/min #misleading in description: r4 and r5 switched (before: r4: 19.68, r5:4.92)
                    release_rate = round(4.92/rho_CH4/60,2) # L/min
                    RR = f'{release_rate} L/min'
                elif peaktime < r5_finish:
                    # 19.68 g/h = 0.5 L/min
                    release_rate = round(19.68/rho_CH4/60,1) # L/min
                    RR = f'{release_rate} L/min'
                else:
                    print('weird mistake')
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/min'
            
        else:
            print(peaktime+'n Day passed not valid, must be either Day1-bike, Day1-car or Day2-car')
            RR = None
            release_rate = None
                
        return RR, release_rate
    
def releaserate_L(peaktime,loca,day):
        # London    
        
        release_rate = None
        RR = '0 L/min'
        release_height = None
        
        if (day == 'Day2'): 
            r1_start = datetime(2019,9,10,10,1)
            r1_finish = datetime(2019,9,10,10,45)
            r2_start = datetime(2019,9,10,11,28)
            r2_finish = datetime(2019,9,10,12,20)
            r3_start = datetime(2019,9,10,12,45) #before: 12:32-12:45 (wrong in pp slide)
            r3_finish = datetime(2019,9,10,13,32)
            r4_start = datetime(2019,9,10,13,58)
            r4_finish = datetime(2019,9,10,14,53)
            r5_start = datetime(2019,9,10,15,0)
            r5_finish = datetime(2019,9,10,15,38)
            r6_start = datetime(2019,9,10,15,40)
            r6_finish = datetime(2019,9,10,16,22)
                    
            peaktime = pd.to_datetime(peaktime)
            
            
            if peaktime >= r1_start:
                if peaktime < r1_finish:
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r2_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r2_finish:
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                elif peaktime < r3_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r3_finish:
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                elif peaktime < r4_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r4_finish:
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r5_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r5_finish:
                    release_rate = 35 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r6_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r6_finish:
                    release_rate = 35 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                else:
                    print('weird mistake')
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/min'
                
        elif (day == 'Day3'): 
            r1_start = datetime(2019,9,11,10,11)
            r1_finish = datetime(2019,9,11,11,1)
            r2_start = datetime(2019,9,11,11,16)
            r2_finish = datetime(2019,9,11,12,5)
            r3_start = datetime(2019,9,11,12,22) #before: 12:32-12:45 (wrong in pp slide)
            r3_finish = datetime(2019,9,11,13,11)
            r4_start = datetime(2019,9,11,13,25)
            r4_finish = datetime(2019,9,11,14,11)
            r5_start = datetime(2019,9,11,14,41)
            r5_finish = datetime(2019,9,11,15,24)
            r6_start = datetime(2019,9,11,15,28)
            r6_finish = datetime(2019,9,11,16,20)
            r7_start = datetime(2019,9,11,16,26)
            r7_finish = datetime(2019,9,11,16,57)
            
                    
            peaktime = pd.to_datetime(peaktime)
            
            
            if peaktime >= r1_start:
                if peaktime < r1_finish: #---------------------
                    release_rate = 35 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                elif peaktime < r2_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r2_finish:  #---------------------
                    release_rate = 35 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r3_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r3_finish: #---------------------
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r4_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r4_finish: #---------------------
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                elif peaktime < r5_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r5_finish: #---------------------
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                elif peaktime < r6_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r6_finish: #---------------------
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r7_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r7_finish: #---------------------
                    release_rate = 35 # L/min # in Excel 25 L/min, but that must be an error (otherwise they only used 35 and 70 L/min)
                    RR = f'{release_rate} L/min'
                    release_height=0
                else:
                    print('weird mistake')
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/min'
                release_height=666
        
        elif (day == 'Day4'): 
            r1_start = datetime(2019,9,12,9,56)
            r1_finish = datetime(2019,9,12,10,36)
            r2_start = datetime(2019,9,12,10,50)
            r2_finish = datetime(2019,9,12,11,40)
            r3_start = datetime(2019,9,12,11,44) #before: 12:32-12:45 (wrong in pp slide)
            r3_finish = datetime(2019,9,12,12,28)
            r4_start = datetime(2019,9,12,12,40)
            r4_finish = datetime(2019,9,12,13,23)
            r5_start = datetime(2019,9,12,13,59)
            r5_finish = datetime(2019,9,12,14,43)
            r6_start = datetime(2019,9,12,14,49)
            r6_finish = datetime(2019,9,12,15,30)
            r7_start = datetime(2019,9,12,15,34)
            r7_finish = datetime(2019,9,12,16,23)
            r8_start = datetime(2019,9,12,16,27)
            r8_finish = datetime(2019,9,12,17,2)
            
                    
            peaktime = pd.to_datetime(peaktime)
            
            
            if peaktime >= r1_start:
                if peaktime < r1_finish:
                    release_rate = 35 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r2_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r2_finish:
                    release_rate = 35 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                elif peaktime < r3_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r3_finish:
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                elif peaktime < r4_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r4_finish:
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r5_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r5_finish:
                    release_rate = 35 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r6_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r6_finish:
                    release_rate = 35 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                elif peaktime < r7_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r7_finish:
                    release_rate = 70 # L/min # in Excel 25 L/min, but that must be an error (otherwise they only used 35 and 70 L/min)
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                elif peaktime < r8_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r8_finish:
                    release_rate = 70 # L/min # in Excel 25 L/min, but that must be an error (otherwise they only used 35 and 70 L/min)
                    RR = f'{release_rate} L/min'
                    release_height=0
                else:
                    print('weird mistake')
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/min'
                release_height=666
        
        elif day == 'Day5': 
            r1_start    = datetime(2019,9,13,9,25)
            r1_finish   = datetime(2019,9,13,10,10)
            r2_start    = datetime(2019,9,13,10,15)
            r2_finish   = datetime(2019,9,13,10,53)
                    
            peaktime = pd.to_datetime(peaktime)
            
            
            if peaktime >= r1_start:
                if peaktime < r1_finish:
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r2_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r2_finish:
                    release_rate = 70 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=3.7
                else:
                    print('weird mistake')
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/min'
        
        else:
            print(f'Day provided not valid (either Day2 or Day5): {day}')
        
        return RR, release_rate, release_height
    
    
def releaserate_L2(peaktime,loca,day):
        # London    
        
        release_rate = None
        RR = '0 L/min'
        release_height = None
        
        if (day == 'Day1'): # time in UTC
            r1_start = datetime(2024,5,13,13,18)
            r1_finish = datetime(2024,5,13,13,44)
            r2_start = datetime(2024,5,13,14,3)
            r2_finish = datetime(2024,5,13,14,31)
            r3_start = datetime(2024,5,13,14,38) 
            r3_finish = datetime(2024,5,13,14,58)
            r4_start = datetime(2024,5,13,15,14)
            r4_finish = datetime(2024,5,13,15,40)
            r5_start = datetime(2024,5,13,15,44)
            r5_finish = datetime(2024,5,13,16,6)
            r6_start = datetime(2024,5,13,16,14)
            r6_finish = datetime(2024,5,13,16,36)
            r7_start = datetime(2024,5,13,16,47) # 16:48 in metadata, but include first peak at 16:47:53 as well?
            r7_finish = datetime(2024,5,13,17,15) # 17:14 in metadata, but include last peak at 17:14:21 as well?
                    
            peaktime = pd.to_datetime(peaktime)
            
            
            if peaktime >= r1_start:
                if peaktime < r1_finish:
                    release_rate = 70.48 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r2_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r2_finish:
                    release_rate = 50.52 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r3_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r3_finish:
                    release_rate = 30.58 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r4_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r4_finish:
                    release_rate = 10.63 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r5_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r5_finish:
                    release_rate = 5.64 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r6_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r6_finish:
                    release_rate = 0.99 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r7_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r7_finish:
                    release_rate = 30.6 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                else:
                    print('weird mistake')         
                    
                    
        elif (day == 'Day2'): # time in UTC
            r1_start = datetime(2024,5,14,9,8)
            r1_finish = datetime(2024,5,14,9,47)
            r2_start = datetime(2024,5,14,10,2)
            r2_finish = datetime(2024,5,14,10,32)
            r3_start = datetime(2024,5,14,10,37)
            r3_finish = datetime(2024,5,14,11,23)
                                    
            peaktime = pd.to_datetime(peaktime)
            
            
            if peaktime >= r1_start:
                if peaktime < r1_finish:
                    release_rate = 0.99 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r2_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r2_finish:
                    release_rate = 0.49 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                elif peaktime < r3_start:
                    release_rate = 0 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=666
                elif peaktime < r3_finish:
                    release_rate = 0.2 # L/min
                    RR = f'{release_rate} L/min'
                    release_height=0
                else:
                    print('weird mistake')
                    
            else: 
                print(f'{peaktime} not in release period')
                release_rate = 0
                RR = '0 L/min'
                  
        else:
            print(f'Day provided not valid (either Day1 or Day2): {day}')
        
        return RR, release_rate
    
    
    
    
def distance(lon_ref, lat_ref, lon, lat):
        R = 6373000.0 # approximate radius of earth in km
        lat1 = np.deg2rad(lat_ref)
        lon1 = np.deg2rad(lon_ref)
        lat2 = np.deg2rad(lat)
        lon2 = np.deg2rad(lon)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = R * c
        
        return distance 
    
    
    
    
def analyse_peak(corrected_peaks,*args): #args*= G43_vars,G23_vars,...
    """
    Main purpose: calculated the integrated spatial peak area per verified peak
    (enhancement) in methane measurements.
    Additionally: some other characteristics (length of transect, mean speed,
    release rates...) are added to the dataframe

    This function processes methane peak data from different measurement instruments
    obtained in different experiments, and stores results in the provided DataFrame.

    Parameters:
    -----------
    corrected_peaks : pandas.DataFrame
        DataFrame containing identified and quality-checked peaks
        The index must be a datetime object.
    
    *args : dict
        Variable number of dictionaries, each representing an instrument's data.
        Each dictionary must contain the following keys:
        - 'df' (pandas.DataFrame): Measurement data with timestamps.
        - 'CH4col' (str): Column name for methane concentration.
        - 'spec' (str): Instrument specification name.
        - 'title' (str): Instrument title used for QC filtering.
        - 'city' (str): City name for selecting the appropriate release rate function.
        - 'day' (str): Measurement day, used in release rate assignment.
    
    Returns:
    --------
    None
        The function modifies `corrected_peaks` in place by adding calculated fields:
        - 'Area_sum_<spec>' : Area under the methane concentration curve using
                            stepwise speed information
        - 'Area_mean_<spec>' : Area estimated using mean speed.
        - 'Max_<spec>' : Maximum methane concentration within the peak.
        - 'Release_rate' : Release rate based on city-specific functions.
        - 'Release_rate_str' : String representation of the release rate.
        - 'Mean_speed' : Average speed of the measurement vehicle per transect/peak
        - 'dx_calc_meanspeed' : Estimated transect length using mean speed.
        - 'dx_gps' : Transect length based on GPS coordinates.
        - 'Longitude', 'Latitude' : Location of the peak.
        - 'Release_height' (only for London data) : Release height.
        - 'QC' : Boolean flag indicating whether the peak passed QC checks.
    
    Notes:
    ------
    - This function assumes that `corrected_peaks` has columns for each instrument
      title used in QC filtering.
    - The function applies city-specific release rate functions based on the `city`
      field in `args`.
    """
    
    
    

    # Define a dictionary mapping city names to releaserate functions
    releaserate_functions = {
        'Rotterdam': releaserate_R,
        'Utrecht': releaserate_U,
        'Toronto': releaserate_T,
        'London': releaserate_L,
        'London_II': releaserate_L2,
        'Utrecht_III': releaserate_U3
        # Add more cities and their corresponding releaserate functions here
        # if city added: update plot_indivpeaks_afterQC function in general_plots
    }
    
    corrected_peaks.index = pd.to_datetime(corrected_peaks.index) # insure index is a datetime object
    
    num_instr = 0
    list_spec = []
    for df_instr in args:
        num_instr += 1
        list_spec.append(df_instr['title'])
        
    
    for df_instr in args:
        
        data       = df_instr['df'].copy() #dum
        data_CH4   = data[df_instr['CH4col']] #y_dum
        spec       = df_instr['spec']
        
        
        print(f'     Looking at {spec} measurement data')
        
        QCcount = 0
        kml     = simplekml.Kml()
        kml_peaks     = simplekml.Kml()
    
        for index1, row in corrected_peaks.iterrows():
            QC=[]
            for i in range(len(list_spec)):
                QC.append(row[list_spec[i]])
            
            
            if all(x == 1.0 for x in QC) and (('GPS' not in row or row['GPS']==1) 
                                              and (('Loc' not in row or row['Loc']!=0))):
                             
                QCcount        += 1
                trans_s         = row['Peakstart_QC']
                trans_e         = row['Peakend_QC']
                peakno          = row['Peak']
                transect_name   = 'Peak '+ str(peakno) + ' ' + spec
                

                data_transect   = data[(data.index >= trans_s)
                                      &(data.index<=trans_e)]
                
                CH4_transect    = data_CH4[(data.index >= trans_s)
                                        &(data.index<=trans_e)]
                
                if 'Loc' in row: # not in London data
                    loca            = row['Loc']
                else:
                    loca = 0
                speed           = data_transect['Speed [m/s]']
                timediff        = (data_transect.index[-1] - data_transect.index[0]).seconds
                meanspeed       = np.mean(data_transect['Speed [m/s]'])
                
                # Call the releaserate function based on the city name
                releaserate_function = releaserate_functions.get(df_instr['city']) # depending on city (and measurment day, choose different releaserate_functions)
                if releaserate_function:
                    if (df_instr['city'] == 'London'):
                        rr_str, rr_fl,r_height  = releaserate_function(index1, loca, df_instr['day'])
                    else:
                        rr_str, rr_fl = releaserate_function(index1, loca, df_instr['day'])
                
                
                # Calculate area with two different methods -----------------
                
                area_method_1      = 0
                dt_CH4_total      = 0
                #area3           = 0 (third method: triangle)
                
                for right_index, row in data_transect.iterrows():
                    if not right_index == data_transect.index[0]:

                        dt          = (right_index - left_index).total_seconds()
                        area_method_1 += dt*CH4_transect[right_index]*speed[right_index]
                        dt_CH4_total += dt*CH4_transect[right_index]
                        
                        
                        #speed_mean   = np.mean([speed[old_index2],speed[index2]])
                        #dc_CH4       = y[index2]-y[old_index2]
                        #area3       += dt*y[old_index2]*speed_mean + dt*0.5*dc_CH4*speed_mean
                        
                    left_index = right_index
                
                #area1       = area_total
                area_method_2       = dt_CH4_total * meanspeed
                
                
                if timediff > 40:
                    print(f'peak {peakno} might be too long')
                    
                index_round = pd.to_datetime(index1)
                
                maxrow = data.iloc[data.index.get_indexer([index_round],method='nearest')[0]] #change get_loc to get_indexer (add [] and [0])
                
                # =============================================================================
                #           Create KML String

                data_transect.reset_index(drop=True, inplace=True)           
                CH4_transect.reset_index(drop=True, inplace=True)
                
                enhancement = (CH4_transect)*10
                lon         = data_transect['Longitude']
                lat         = data_transect['Latitude']
                coord       = []
                
                for k in range(len(lon)):                                                                      
                    p = (lon[k],lat[k],enhancement[k])
                    coord.append(p)

                # Calculate length of transects ------------------------------------
                
                # 1. Method ---
                # dx_gps                  = distance(coord[0][0],coord[0][1],
                #                                    coord[-1][0],coord[-1][1]) # last coord - first coord (only working for straight drives)
                dx_gps = 0

                for i in range(len(coord)-1):
                    lon1, lat1, _ = coord[i]
                    lon2, lat2, _ = coord[i+1]
                    coords_1 = (lat1, lon1)
                    coords_2 = (lat2, lon2)
                    #dist = distance(coords_1, coords_2).meters
                    dist = distance(lon1, lat1, lon2, lat2)
                    dx_gps += dist
                    
                # 2. Method ---
                dx_calc_meanspeed       = meanspeed * timediff

                # ---
                linestring              = kml.newlinestring(name=transect_name)
                linestring.coords       = coord
                linestring.altitudemode = simplekml.AltitudeMode.relativetoground
                kml_peaks.newpoint(name=f"Peak {peakno}", coords=[(maxrow['Longitude'],maxrow['Latitude'])])
    # =============================================================================
        
                corrected_peaks.loc[index1, f'Area_sum_{spec}']   = area_method_1
                corrected_peaks.loc[index1, f'Area_mean_{spec}']  = area_method_2
                #corrected_peaks.loc[index1, f'Area_{spec}_triangle']  = area3
                corrected_peaks.loc[index1, 'Release_rate']      = rr_fl
                corrected_peaks.loc[index1, 'Release_rate_str']  = rr_str
                corrected_peaks.loc[index1, f'Max_{spec}']       = max(CH4_transect)
                corrected_peaks.loc[index1, 'QC']                = True
                corrected_peaks.loc[index1, 'Mean_speed']        = meanspeed
                corrected_peaks.loc[index1, 'dx_calc_meanspeed'] = dx_calc_meanspeed
                corrected_peaks.loc[index1, 'dx_gps']            = dx_gps
                corrected_peaks.loc[index1, 'Longitude']         = maxrow['Longitude']
                corrected_peaks.loc[index1, 'Latitude']          = maxrow['Latitude']
                if df_instr['city'] == 'London':
                    corrected_peaks.loc[index1, 'Release_height']    = r_height
                if rr_fl == 0:
                    corrected_peaks.loc[index1, 'QC']                = False
            else:
                corrected_peaks.loc[index1, 'QC']                = False
                
            
        #kml.save(path_res +'KMLs/U_'+ name + ".kml")
        #kml_peaks.save(path_res +'KMLs/U_Peaks_QCpassed.kml')
        print(f'         {spec} done,',  QCcount, 'good transects')
        
        
        
def analyse_peak_U3_old(corrected_peaks,*args): #args*= G43_vars,G23_vars,...

    ''' due to interpolation of gps during merging of gps with CH4 data, the original function
    built to deal with the periodes with no gps data does not work anymore'''

    # Define a dictionary mapping city names to releaserate functions
    releaserate_functions = {
        'Rotterdam': releaserate_R,
        'Utrecht': releaserate_U,
        'Toronto': releaserate_T,
        'London': releaserate_L,
        'London_II': releaserate_L2,
        'Utrecht_III': releaserate_U3
        # Add more cities and their corresponding releaserate functions here
        # if city added: update plot_indivpeaks_afterQC function in general_plots
    }
    
    corrected_peaks.index = pd.to_datetime(corrected_peaks.index) # insure index is a datetime object
    
    num_instr = 0
    list_spec = []
    for df_instr in args:
        num_instr += 1
        list_spec.append(df_instr['title'])
        
    
    for df_instr in args:
        
        data       = df_instr['df'].copy() #dum
        data_CH4   = data[df_instr['CH4col']] #y_dum
        spec       = df_instr['spec']
        
        
        print(f'     Looking at {spec} measurement data')
        
        QCcount = 0
        # kml     = simplekml.Kml()
        # kml_peaks     = simplekml.Kml()
        print(df_instr['title'])
        for index1, row in corrected_peaks.iterrows():
            
            QC=[]
            for i in range(len(list_spec)):
                QC.append(row[list_spec[i]])
                
            if all(x == 0 for x in QC):
                corrected_peaks.loc[index1, 'QC']  = False
            elif((row['GPS']==0) or ((row['Loc']==0))):
                corrected_peaks.loc[index1, 'QC']  = False
            else:    
                corrected_peaks.loc[index1, 'QC']  = True
            
            
            if row[df_instr['title']]==1 and (('GPS' not in row or row['GPS']==1 or row['GPS']==2) 
                                              and (('Loc' not in row or row['Loc']!=0))):
                 
                QCcount        += 1
                trans_s         = row['Peakstart_QC']
                trans_e         = row['Peakend_QC']
                peakno          = row['Peak']
                transect_name   = 'Peak '+ str(peakno) + ' ' + spec

                data_transect   = data[(data.index >= trans_s)
                                      &(data.index<=trans_e)]
                
                CH4_transect    = data_CH4[(data.index >= trans_s)
                                        &(data.index<=trans_e)]
                
                if 'Loc' in row: # not in London data
                    loca            = row['Loc']
                else:
                    loca = 0
                speed           = data_transect['Speed [m/s]']
                timediff        = (data_transect.index[-1] - data_transect.index[0]).seconds
                meanspeed       = np.mean(data_transect['Speed [m/s]'])
                
                # Call the releaserate function based on the city name
                releaserate_function = releaserate_functions.get(df_instr['city']) # depending on city (and measurment day, choose different releaserate_functions)
                if releaserate_function:
                    if (df_instr['city'] == 'London'):
                        rr_str, rr_fl,r_height  = releaserate_function(index1, loca, df_instr['day'])
                    else:
                        rr_str, rr_fl = releaserate_function(index1, loca, df_instr['day'])
                
                
                # Calculate area with two different methods -----------------
                
                # Case 1: gps information is not available (or only partly, indicated by GPS=2 in the QC)
                if row['GPS']==2:
                    
                    
                    speed_default = 6.9 # mean speed of transects where gps is available
                    
                    meanspeed = np.mean(data_transect['Speed [m/s]'].copy().dropna().values)
                    if meanspeed == 0:
                        raise Exception('Nul mean speed error')
                    if pd.isna(meanspeed):
                        meanspeed = speed_default
                        
                        
                    area_method_1 = 0
                    area_method_2 = 0
                    
                    method_1_proportion_count = []
                    
                    for right_index, row in data_transect.iterrows():
                        if not right_index==data_transect.index[0]:
                            
                            dt = (right_index - left_index).total_seconds()
                            if not pd.isna(row['Speed [m/s]']):
                                area_method_1 += dt*CH4_transect[right_index]*speed[right_index]
                                method_1_proportion_count.append(1)
                        
                            else:
                                method_1_proportion_count.append(0)
                                area_method_1 += dt*CH4_transect[right_index]*meanspeed
                                
                            area_method_2 += dt*CH4_transect[right_index]*meanspeed
                            
                        left_index = right_index  
                        
                    method_1_clean = sum(method_1_proportion_count) / len(method_1_proportion_count) * 100
                    
                        
                        
                # Case 2: gps information is available throughout the whole transect  
                else:   
                    area_method_1      = 0
                    dt_CH4_total      = 0
                    test = 0
                    #area3           = 0 (third method: triangle)
                    
                    for right_index, row in data_transect.iterrows():
                        if not right_index == data_transect.index[0]:
    
                            dt          = (right_index - left_index).total_seconds()
                            area_method_1 += dt*CH4_transect[right_index]*speed[right_index]
                            dt_CH4_total += dt*CH4_transect[right_index]
                            test += dt*CH4_transect[right_index]*meanspeed
                            
                            
                            #speed_mean   = np.mean([speed[old_index2],speed[index2]])
                            #dc_CH4       = y[index2]-y[old_index2]
                            #area3       += dt*y[old_index2]*speed_mean + dt*0.5*dc_CH4*speed_mean
                            
                        left_index = right_index
                    
                    #area1       = area_total
                    area_method_2       = dt_CH4_total * meanspeed
                    print(test)
                    print(area_method_2)
                    method_1_clean = 100
                
                if timediff > 40:
                    print(f'peak {peakno} might be too long')
                    
                index_round = pd.to_datetime(index1)
                
                maxrow = data.iloc[data.index.get_indexer([index_round],method='nearest')[0]] #change get_loc to get_indexer (add [] and [0])
                
                peaklat = maxrow['Latitude']
                peaklon = maxrow['Longitude']
    # =============================================================================
    #           Create KML String

                data_transect.reset_index(drop=True, inplace=True)           
                CH4_transect.reset_index(drop=True, inplace=True)
                
                enhancement = (CH4_transect)*10
                lon         = data_transect['Longitude']
                lat         = data_transect['Latitude']
                coord       = []
                
                for k in range(len(lon)):                                                                      
                    p = (lon[k],lat[k],enhancement[k])
                    coord.append(p)

                # Calculate length of transects ------------------------------------
                
                # 1. Method ---
                # dx_gps                  = distance(coord[0][0],coord[0][1],
                #                                    coord[-1][0],coord[-1][1]) # last coord - first coord (only working for straight drives)
                dx_gps = 0

                for i in range(len(coord)-1):
                    lon1, lat1, _ = coord[i]
                    lon2, lat2, _ = coord[i+1]
                    coords_1 = (lat1, lon1)
                    coords_2 = (lat2, lon2)
                    #dist = distance(coords_1, coords_2).meters
                    dist = distance(lon1, lat1, lon2, lat2)
                    dx_gps += dist
                    
                # 2. Method ---
                dx_calc_meanspeed       = meanspeed * timediff

                # ---
                # linestring              = kml.newlinestring(name=transect_name)
                # linestring.coords       = coord
                # linestring.altitudemode = simplekml.AltitudeMode.relativetoground
                # kml_peaks.newpoint(name=f"Peak {peakno}", coords=[(peaklon,peaklat)])
    # =============================================================================
        
                corrected_peaks.loc[index1, f'Area_sum_{spec}']   = area_method_1
                corrected_peaks.loc[index1, f'Area_mean_{spec}']  = area_method_2
                #corrected_peaks.loc[index1, f'Area_{spec}_triangle']  = area3
                corrected_peaks.loc[index1, 'Release_rate']      = rr_fl
                corrected_peaks.loc[index1, 'Release_rate_str']  = rr_str
                corrected_peaks.loc[index1, f'Max_{spec}']       = max(CH4_transect)
                # corrected_peaks.loc[index1, 'QC']                = True
                corrected_peaks.loc[index1, 'Mean_speed']        = meanspeed
                corrected_peaks.loc[index1, 'dx_calc_meanspeed'] = dx_calc_meanspeed
                corrected_peaks.loc[index1, 'dx_gps']            = dx_gps
                corrected_peaks.loc[index1, 'Longitude']         = peaklon
                corrected_peaks.loc[index1, 'Latitude']          = peaklat
                if df_instr['city'] == 'London':
                    corrected_peaks.loc[index1, 'Release_height']    = r_height
                if df_instr['city'] == 'Utrecht_III':
                    corrected_peaks.loc[index1, 'Proportion of mean speed in Method 1 [%]']    = method_1_clean
                if rr_fl == 0:
                    corrected_peaks.loc[index1, 'QC']                = False
            # else:
            #     corrected_peaks.loc[index1, 'QC']                = False
                
            
        #kml.save(path_res +'KMLs/U_'+ name + ".kml")
        #kml_peaks.save(path_res +'KMLs/U_Peaks_QCpassed.kml')
        print(f'         {spec} done,',  QCcount, 'good transects')
        

def analyse_peak_U3(corrected_peaks,*args): #args*= G43_vars,G23_vars,...

    # Define a dictionary mapping city names to releaserate functions
    releaserate_functions = {
        'Rotterdam': releaserate_R,
        'Utrecht': releaserate_U,
        'Toronto': releaserate_T,
        'London': releaserate_L,
        'London_II': releaserate_L2,
        'Utrecht_III': releaserate_U3
        # Add more cities and their corresponding releaserate functions here
        # if city added: update plot_indivpeaks_afterQC function in general_plots
    }
    
    corrected_peaks.index = pd.to_datetime(corrected_peaks.index) # insure index is a datetime object
    
    num_instr = 0
    list_spec = []
    for df_instr in args:
        num_instr += 1
        list_spec.append(df_instr['title'])
        
    
    for df_instr in args:
        
        data       = df_instr['df'].copy() #dum
        data_CH4   = data[df_instr['CH4col']] #y_dum
        spec       = df_instr['spec']
        
        
        print(f'     Looking at {spec} measurement data')
        
        QCcount = 0
        # kml     = simplekml.Kml()
        # kml_peaks     = simplekml.Kml()
        print(df_instr['title'])
        for index1, row in corrected_peaks.iterrows():
            
            QC=[]
            for i in range(len(list_spec)):
                QC.append(row[list_spec[i]])
                
            if all(x == 0 for x in QC):
                corrected_peaks.loc[index1, 'QC']  = False
            elif((row['GPS']==0) or ((row['Loc']==0))):
                corrected_peaks.loc[index1, 'QC']  = False
            else:    
                corrected_peaks.loc[index1, 'QC']  = True
            
            
            if row[df_instr['title']]==1 and (('GPS' not in row or row['GPS']==1 or row['GPS']==2) 
                                              and (('Loc' not in row or row['Loc']!=0))):
                 
                QCcount        += 1
                trans_s         = row['Peakstart_QC']
                trans_e         = row['Peakend_QC']
                peakno          = row['Peak']
                transect_name   = 'Peak '+ str(peakno) + ' ' + spec

                data_transect   = data[(data.index >= trans_s)
                                      &(data.index<=trans_e)]
                
                CH4_transect    = data_CH4[(data.index >= trans_s)
                                        &(data.index<=trans_e)]
                
                if 'Loc' in row: # not in London data
                    loca            = row['Loc']
                else:
                    loca = 0
                speed           = data_transect['Speed [m/s]']
                timediff        = (data_transect.index[-1] - data_transect.index[0]).seconds
                meanspeed       = np.mean(data_transect['Speed [m/s]'])
                
                # Call the releaserate function based on the city name
                releaserate_function = releaserate_functions.get(df_instr['city']) # depending on city (and measurment day, choose different releaserate_functions)
                if releaserate_function:
                    if (df_instr['city'] == 'London'):
                        rr_str, rr_fl,r_height  = releaserate_function(index1, loca, df_instr['day'])
                    else:
                        rr_str, rr_fl = releaserate_function(index1, loca, df_instr['day'])
                
                
                # Calculate area with two different methods -----------------
                
                # Case 1: gps information is not available (or only partly, indicated by GPS=2 in the QC)
                if row['GPS']==2:
                    print(row['GPS'])
                    
                    meanspeed = 5.9 # mean & median of all Mean_speed for peaks where gps is available is 5.9 m/s
                    # Mean = 5.941; 0.25 quantile = 5.291; Median = 5.933; 0.75 quantile = 6.623
                        
                    area_method_1 = 0 # will stay 0, since no gps data available
                    area_method_2 = 0 # defaut mean speed value used to calculate area method 2
                    
                    
                    for right_index, row in data_transect.iterrows():
                        if not right_index==data_transect.index[0]:
                            
                            dt = (right_index - left_index).total_seconds()
                            area_method_2 += dt*CH4_transect[right_index]*meanspeed
                            
                        left_index = right_index  
                        
                        
                # Case 2: gps information is available throughout the whole transect  
                else:   
                    area_method_1      = 0
                    area_method_2      = 0
                    #area3           = 0 (third method: triangle)
                    
                    for right_index, row in data_transect.iterrows():
                        if not right_index == data_transect.index[0]:
    
                            dt          = (right_index - left_index).total_seconds()
                            area_method_1 += dt*CH4_transect[right_index]*speed[right_index]
                            area_method_2 += dt*CH4_transect[right_index]*meanspeed
                            
                            #speed_mean   = np.mean([speed[old_index2],speed[index2]])
                            #dc_CH4       = y[index2]-y[old_index2]
                            #area3       += dt*y[old_index2]*speed_mean + dt*0.5*dc_CH4*speed_mean
                            
                        left_index = right_index
                    
                
                if timediff > 40:
                    print(f'peak {peakno} might be too long')
                    
                index_round = pd.to_datetime(index1)
                
                maxrow = data.iloc[data.index.get_indexer([index_round],method='nearest')[0]] #change get_loc to get_indexer (add [] and [0])
                
                peaklat = maxrow['Latitude']
                peaklon = maxrow['Longitude']
    # =============================================================================
    #           Create KML String

                data_transect.reset_index(drop=True, inplace=True)           
                CH4_transect.reset_index(drop=True, inplace=True)
                
                enhancement = (CH4_transect)*10
                lon         = data_transect['Longitude']
                lat         = data_transect['Latitude']
                coord       = []
                
                for k in range(len(lon)):                                                                      
                    p = (lon[k],lat[k],enhancement[k])
                    coord.append(p)

                # Calculate length of transects ------------------------------------
                
                # 1. Method ---
                # dx_gps                  = distance(coord[0][0],coord[0][1],
                #                                    coord[-1][0],coord[-1][1]) # last coord - first coord (only working for straight drives)
                dx_gps = 0

                for i in range(len(coord)-1):
                    lon1, lat1, _ = coord[i]
                    lon2, lat2, _ = coord[i+1]
                    coords_1 = (lat1, lon1)
                    coords_2 = (lat2, lon2)
                    #dist = distance(coords_1, coords_2).meters
                    dist = distance(lon1, lat1, lon2, lat2)
                    dx_gps += dist
                    
                # 2. Method ---
                dx_calc_meanspeed       = meanspeed * timediff

                # ---
                # linestring              = kml.newlinestring(name=transect_name)
                # linestring.coords       = coord
                # linestring.altitudemode = simplekml.AltitudeMode.relativetoground
                # kml_peaks.newpoint(name=f"Peak {peakno}", coords=[(peaklon,peaklat)])
    # =============================================================================
        
                corrected_peaks.loc[index1, f'Area_sum_{spec}']   = area_method_1
                corrected_peaks.loc[index1, f'Area_mean_{spec}']  = area_method_2
                #corrected_peaks.loc[index1, f'Area_{spec}_triangle']  = area3
                corrected_peaks.loc[index1, 'Release_rate']      = rr_fl
                corrected_peaks.loc[index1, 'Release_rate_str']  = rr_str
                corrected_peaks.loc[index1, f'Max_{spec}']       = max(CH4_transect)
                # corrected_peaks.loc[index1, 'QC']                = True
                corrected_peaks.loc[index1, 'Mean_speed']        = meanspeed
                corrected_peaks.loc[index1, 'dx_calc_meanspeed'] = dx_calc_meanspeed
                corrected_peaks.loc[index1, 'dx_gps']            = dx_gps
                corrected_peaks.loc[index1, 'Longitude']         = peaklon
                corrected_peaks.loc[index1, 'Latitude']          = peaklat
                if df_instr['city'] == 'London':
                    corrected_peaks.loc[index1, 'Release_height']    = r_height
                if rr_fl == 0:
                    corrected_peaks.loc[index1, 'QC']                = False
            # else:
            #     corrected_peaks.loc[index1, 'QC']                = False
                
            
        #kml.save(path_res +'KMLs/U_'+ name + ".kml")
        #kml_peaks.save(path_res +'KMLs/U_Peaks_QCpassed.kml')
        print(f'         {spec} done,',  QCcount, 'good transects')
        
        
def analyse_peak_withoutU3(corrected_peaks,*args): #args*= G43_vars,G23_vars,...

    # Define a dictionary mapping city names to releaserate functions
    releaserate_functions = {
        'Rotterdam': releaserate_R,
        'Utrecht': releaserate_U,
        'Toronto': releaserate_T,
        'London': releaserate_L,
        'London_II': releaserate_L2,
        'Utrecht_III': releaserate_U3
        # Add more cities and their corresponding releaserate functions here
    }
    
    
    num_instr = 0
    list_spec = []
    for df_instr in args:
        num_instr += 1
        list_spec.append(df_instr['title'])
        
    
    for df_instr in args:
        
        data       = df_instr['df'].copy() #dum
        data_CH4   = data[df_instr['CH4col']] #y_dum
        spec       = df_instr['spec']
        
        
        print(f'     Looking at {spec} measurement data')
        
        QCcount = 0
        kml     = simplekml.Kml()
        kml_peaks     = simplekml.Kml()
    
        for index1, row in corrected_peaks.iterrows():
            QC=[]
            for i in range(len(list_spec)):
                QC.append(row[list_spec[i]])
            
            
            if all(x == 1.0 for x in QC) and (('GPS' not in row or row['GPS']==1) 
                                              and (('Loc' not in row or row['Loc']!=0))):
            #if (row['QC']==True):
                 
                QCcount        += 1
                trans_s         = row['Peakstart_QC']
                trans_e         = row['Peakend_QC']
                peakno          = row['Peak']
                transect_name   = 'Peak '+ str(peakno) + ' ' + spec

                dat             = data[(data.index >= trans_s)
                                      &(data.index<=trans_e)]
                
                y               = data_CH4[(data.index >= trans_s)
                                        &(data.index<=trans_e)]
                
                if 'Loc' in row: # not in London data
                    loca            = row['Loc']
                else:
                    loca = 0
                speed           = dat['Speed [m/s]']
                timediff        = (dat.index[-1] - dat.index[0]).seconds
                meanspeed       = np.mean(dat['Speed [m/s]'])
                
                # Call the releaserate function based on the city name
                releaserate_function = releaserate_functions.get(df_instr['city']) # depending on city (and measurment day, choose different releaserate_functions)
                if releaserate_function:
                    if (df_instr['city'] == 'London'):
                        rr_str, rr_fl,r_height  = releaserate_function(index1, loca, df_instr['day'])
                    else:
                        rr_str, rr_fl = releaserate_function(index1, loca, df_instr['day'])
                
                
                # Calculate area with two different methods
                area_total      = 0
                dt_y_total      = 0
                #area3           = 0 (third method: triangle)
                
                for index2, row in dat.iterrows():
                    if not index2 == dat.index[0]:

                        dt          = (index2 - old_index2).total_seconds()
                        area_total += dt*y[index2]*speed[index2]
                        dt_y_total += dt*y[index2]
                        
                        
                        #speed_mean   = np.mean([speed[old_index2],speed[index2]])
                        #dc_CH4       = y[index2]-y[old_index2]
                        #area3       += dt*y[old_index2]*speed_mean + dt*0.5*dc_CH4*speed_mean
                        
                    old_index2 = index2
                
                area1       = area_total
                area2       = dt_y_total * meanspeed
                
                if timediff > 40:
                    print(f'peak {peakno} might be too long')
                    
                index_round = pd.to_datetime(index1)
                
                maxrow = data.iloc[data.index.get_indexer([index_round],method='nearest')[0]] #change get_loc to get_indexer (add [] and [0])
                
                peaklat = maxrow['Latitude']
                peaklon = maxrow['Longitude']
    # =============================================================================
    #           Create KML String

                dat.reset_index(drop=True, inplace=True)           
                y.reset_index(drop=True, inplace=True)
                
                enhancement = (y)*10
                lon         = dat['Longitude']
                lat         = dat['Latitude']
                coord       = []
                
                for k in range(len(lon)):                                                                      
                    p = (lon[k],lat[k],enhancement[k])
                    coord.append(p)

                # Calculate length of transects ------------------------------------
                
                # 1. Method ---
                # dx_gps                  = distance(coord[0][0],coord[0][1],
                #                                    coord[-1][0],coord[-1][1]) # last coord - first coord (only working for straight drives)
                dx_gps = 0

                for i in range(len(coord)-1):
                    lon1, lat1, _ = coord[i]
                    lon2, lat2, _ = coord[i+1]
                    coords_1 = (lat1, lon1)
                    coords_2 = (lat2, lon2)
                    #dist = distance(coords_1, coords_2).meters
                    dist = distance(lon1, lat1, lon2, lat2)
                    dx_gps += dist
                    
                # 2. Method ---
                dx_calc_meanspeed       = meanspeed * timediff

                # ---
                linestring              = kml.newlinestring(name=transect_name)
                linestring.coords       = coord
                linestring.altitudemode = simplekml.AltitudeMode.relativetoground
                kml_peaks.newpoint(name=f"Peak {peakno}", coords=[(peaklon,peaklat)])
    # =============================================================================
        
                corrected_peaks.loc[index1, f'Area_sum_{spec}']   = area1
                corrected_peaks.loc[index1, f'Area_mean_{spec}']  = area2
                #corrected_peaks.loc[index1, f'Area_{spec}_triangle']  = area3
                corrected_peaks.loc[index1, 'Release_rate']      = rr_fl
                corrected_peaks.loc[index1, 'Release_rate_str']  = rr_str
                corrected_peaks.loc[index1, f'Max_{spec}']       = max(y)
                corrected_peaks.loc[index1, 'QC']                = True
                corrected_peaks.loc[index1, 'Mean_speed']        = meanspeed
                corrected_peaks.loc[index1, 'dx_calc_meanspeed'] = dx_calc_meanspeed
                corrected_peaks.loc[index1, 'dx_gps']            = dx_gps
                corrected_peaks.loc[index1, 'Longitude']         = peaklon
                corrected_peaks.loc[index1, 'Latitude']          = peaklat
                if df_instr['city'] == 'London':
                    corrected_peaks.loc[index1, 'Release_height']    = r_height
                if rr_fl == 0:
                    corrected_peaks.loc[index1, 'QC']                = False
            else:
                corrected_peaks.loc[index1, 'QC']                = False
                
            
        #kml.save(path_res +'KMLs/U_'+ name + ".kml")
        #kml_peaks.save(path_res +'KMLs/U_Peaks_QCpassed.kml')
        print(f'         {spec} done,',  QCcount, 'good transects')
        
        
        
      
def add_distance_to_df(df,city,Day=None):
    # Release Coordinates
    release_L_loc1 = (52.233343, -0.437888)
    release_L2_loc1 = (52.23438, -0.44161)
    release_R_loc1 = (51.9201216, 4.5237450)
    release_R_loc2 = (51.9203931, 4.5224917)
    release_R_loc3 = (51.921028, 4.523775) 
    release_U_loc1 = (52.0874256, 5.1647191) #? (maybe 52.08850892,5.16532755 and 52.08860602,5.16401192)
    release_U_loc2 = (52.0885635, 5.1644029) #?
    release_U3_loc1_1 = (52.0874472, 5.164652777777778) 
    release_U3_loc1_2 = (52.0875333, 5.16506388888889)
    release_U3_loc2 = (52.0885333, 5.164452777777778)
    release_T_loc1 = (43.655007, -79.325254)
    release_T_loc2 = (43.782970, -79.46952)
    
    R_dict = {
        1: release_R_loc1,
        2: release_R_loc2,
        3: release_R_loc3
        }
    U_dict = {
        1: release_U_loc1,
        2: release_U_loc2
        }
    U3_dict = {
        1: release_U3_loc1_2,
        10: release_U3_loc1_2,
        2: release_U3_loc2,
        20: release_U3_loc2,
        3: release_U3_loc1_1,
        30: release_U3_loc1_1
        }
    
    df.reset_index(inplace=True,drop=False)
    
    if (city == 'Rotterdam'):
        distances = []
        for i in range(len(df)):
            release_loc = R_dict[df.loc[i, 'Loc']]
            x = geodesic(release_loc, (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])).meters
            distances.append(x)
        df['Distance_to_source'] = distances
        
    elif (city == 'Utrecht'):
        distances = []
        for i in range(len(df)):
            release_loc = U_dict[df.loc[i, 'Loc']]
            x = geodesic(release_loc, (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])).meters
            distances.append(x)
        df['Distance_to_source'] = distances
    
    elif (city == 'Utrecht_III'):
        distances = []
        for i in range(len(df)):
            release_loc = U3_dict[df.loc[i, 'Loc']]
            if not pd.isna(df.loc[i, 'Latitude']):
                x = geodesic(release_loc, (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])).meters
            else:
                x=0 # coordinates not available for all timeframes
            distances.append(x)
        df['Distance_to_source'] = distances
        
    elif city == 'Toronto':
        distances = []
        if Day == 1:
            release_loc = release_T_loc1
        elif Day == 2:
            release_loc = release_T_loc2
        else: print('Toronto: wrong day')         
        for i in range(len(df)):
            x = geodesic(release_loc, (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])).meters
            distances.append(x)           
        df['Distance_to_source'] = distances
        
    elif city == 'London':
        distances = []
        release_loc = release_L_loc1
        for i in range(len(df)):
            x = geodesic(release_loc, (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])).meters
            distances.append(x)           
        df['Distance_to_source'] = distances
        
    elif city == 'London_II':
        distances = []
        release_loc = release_L2_loc1
        for i in range(len(df)):
            x = geodesic(release_loc, (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])).meters
            distances.append(x)           
        df['Distance_to_source'] = distances
                
    else: print('Wrong city')
    
    df.set_index('Datetime', inplace=True)
    return df
        

def combine_columns(row):
    return (row['Loc'], row['Release_rate'])

        
        
        
        
        
        
        