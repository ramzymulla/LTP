#!/usr/bin/env python
######################################################################################################################
# LTP.py
# Description: This is a python module for loading and analyzing .csv raw data files obtained the MED64 probe. 
# It takes in the relevant data and packages it in a LTP class object, which includes the basic fEPSP measures.
# The actual data analysis and figure generation should be done in the command-line, using a separate script, or in a
# jupyter notebook (make sure to use '%matplotlib widget' or '%matplotlib notebook' to make high resolution figures). 
######################################################################################################################

__version__ = "1.0"
__author__ = "Ramzy Al-Mulla"

import os
import re
import pandas as pd
import numpy as np
import scipy as sp
import csv
import pickle as pl
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt


'''
Globals: These are vital information regarding settings on the MED64/mobius at the time of collection. 
This program will break if any of these are changed without updating the corresponding global!
'''
LT = 100                    # length of each trace in milliseconds (LT -> "Length of Traces")
ST = 0.05                   # time between each sample in milliseconds (ST -> "Sampling Time")
NS = int(LT/ST)             # number of samples per trace (NS -> "Number of Samples")
TS = 5                      # time of stim in milliseconds (TS -> "Time of Stimulus")
SA = int((TS+0.5)/ST)       # sample number where the stim artifact has ended (SA -> "Stimulus Artifact")
SP = 20                     # period between steps in seconds (SP -> "Step Period")
NC = 64                     # number of channels (NC -> "Number of Channels")
SCALE = 1000                # set SCALE = 1 for millivolts, SCALE = 1000 for microvolts
MU = 'μ'
BASLINElabel = "Baseline"
TBSlabel = "Post-Tetanus"        
LTPlabel = "Early LTP"      # If data was only recorded to 60 minutes post TBS, then it is Early LTP data. 
                                # If data extends to 90+ minuts post TBS, you have regular LTP. 

TBScolor = "#D73F09"        # figure color scheme
BASELINEcolor = "#000000"
POST45color = "#00859B"
SLOPEcolor = "#D73F09"
MINcolor = "#00859B"
PEAKcolor = "#AA9D2E"
FVcolor = "#FFB500"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class LTP:
    '''
    LTP Class:
    This is an object for storing the raw dataset, along with the corresponding metadata (ie. stim channels, stim
    strengths, etc.).
    '''
    def __init__(self, ID: str,
                 filepath: str,
                 df: pd.DataFrame,
                 forward: tuple[int, int],
                 reverse: tuple[int, int],
                 hemi = '',
                 stims = (None, None),
                 blind = False,
                 full = False):
        '''
        Initializer for LTP objects
        Args:
            ID: mouse identifier (ie. "B22R", "H22L_LHS1")
            filepath: full path to raw data file
            df: pandas dataframe containing all of the data from the .csv
            forward: Tuple -> (fw stim channel #, fw response channel #)
            reverse: Tuple -> (rev stim channel #, rev response channel #)
            stims*: Tuple -> (fw stim strength (µA), rev stim strength (µA))
        *NOTE: the stim strengths are nonessential information, will be set to "None" if they are not provided
        '''

        self.ID = ID                                                        # mouse ID
        self.hemi = hemi
        self.filepath = filepath                                            # filepath for troubleshooting purposes

        df.columns = [i.strip() for i in df.columns]                        # mobius puts a random space (" ") in each
                                                                            #       channel heading for some reason .-.
        self.df = df                                                        # dataframe containing raw data
        if full:
            self.full = self.generate_full()

        if not blind:
            self.stim_strengths = {'fw':stims[0],                               # dictionary of stimulation strengths (µA)
                                   'rev':stims[1]}
            self.fw_s = forward[0]                                              # forward stim channel
            self.fw_r = forward[1]                                              # forward response channel
            self.rev_s = reverse[0]                                             # reverse stim channel
            self.rev_r = reverse[1]                                             # reverse response channel

            self.traces = self.generate_traces()                                # generate traces
            self.num_traces = len(self.traces)                                  # store number of traces
            self.fw = self.traces[::2].copy()     # forward traces
            self.rev = self.traces[1::2].copy()   # reverse traces


            self.fwmin = (np.min([i.min[2] for i in self.fw]),
                          np.argmin([i.min[2] for i in self.fw]))
            self.revmin = (min([i.min[2] for i in self.rev]),
                           np.argmin([i.min[2] for i in self.rev]))

            self.timestamps = self.generate_timestamps()                        # timestamp (minutes) arrays for each step
            self.tet_time = self.find_tet_time()                                # time of tetanus (minutes, trace #)
            self.timeline, self.timeline_single = self.generate_timeline()
            self.fwdict, self.revdict = self.generate_trace_dicts()
            self.fwstats, self.revstats, self.statlabels = self.generate_stats()


    def __str__(self):
        if self.stim_strengths['fw'] is None:
            return f"{self.ID}:\n " \
                    f"Forward path is {self.fw_s}->{self.fw_r}\n " \
                    f"Reverse path is {self.rev_s}->{self.rev_r}\n " \
                    f"Tetanus occurred at {self.tet_time[0]} " \
                    f"minutes (trace #{self.tet_time[1]+1})\n " \
                    f"Filepath: {self.filepath}\n"
        else:
            return f"{self.ID}:\n " \
                    f"Forward path is {self.fw_s}->{self.fw_r} | {self.stim_strengths['fw']} µA\n " \
                    f"Reverse path is {self.rev_s}->{self.rev_r} | {self.stim_strengths['rev']} µA\n " \
                    f"Tetanus occurred at {self.tet_time[0]} " \
                    f"minutes (trace #{self.tet_time[1]+1}" \
                    f", step 1-{self.tet_time[2]+1})\n " \
                    f"Filepath: {self.filepath}\n"

    def pickle_df(self, subfolder = '',overwrite=False):
        '''
        Pickles this mouse's dataframe to allow for **MUCH** faster load times for subsequent runs
        Args:
            subfolder: name of subfolder to save the file in (defaults to current working directory)
        Returns:
            {Self.ID}_df.pkl in
        '''
        os.makedirs(os.path.join(subfolder,'Pickles'), exist_ok=True)
        path = os.path.join(subfolder,'Pickles')
        # fname_init = os.path.split(self.filepath)[1].split('.')[0].split('_')
        # fname = ''
        # for i in fname_init:
        #     if i.lower() != 'raw':
        #         fname += i+'_'
        # fpath = os.path.join(path,f"{fname}df.pkl")

        fpath = os.path.join(path,f"{self.ID}_df.pkl")
        
        if overwrite or not os.path.exists(fpath):
            self.df.to_pickle(fpath)
        else:
            print("Pickle already exists. If you would like to overwrite, please rerun with overwrite = TRUE")
        
    def generate_traces(self) -> list:
        '''
        This function generates a list of Trace objects from the raw dataframe's fw and rev channels
        Returns:
            list of Trace objects
        '''
        fw_rec = f"ch{self.fw_r}_mV"
        rev_rec = f"ch{self.rev_r}_mV"
        fw_arr = list(self.df[fw_rec])                                      # allocate data arrays
        rev_arr = list(self.df[rev_rec])
        N = len(fw_arr)
        trace_arr = []                                                      # allocate trace array
        ind = 0                                                             # initialize index counter
        counter = 1                                                         # initialize trace counter
        while ind + NS <= N:
            if counter%2 == 1:                                              # alternate between fw and rev
                trace_arr.append(Trace(fw_arr[ind:ind+NS].copy(), counter, self.fw_r, self.ID))
            else:
                trace_arr.append(Trace(rev_arr[ind:ind+NS].copy(), counter, self.rev_r, self.ID))
            counter += 1                                                    # increment counter
            ind += NS                                                       # move to next trace
        return trace_arr

    def generate_full(self):
        '''
        Generates a dictionary containing lists of traces for all channels
        '''
        trace_dict = {}
        for i in range(1,NC+1):
            trace_dict[f"ch{i}_mV"] = []

        N = len(self.df["ch1_mV"])
        ind = 0
        counter = 1
        while ind+NS <= N:
            for channel in trace_dict:
                trace_dict[channel].append(Trace(self.df[channel][ind:ind+NS], counter,
                                                 int(channel[2]), self.ID, exp = False))
            counter += 1
            ind += NS
        return trace_dict

    def generate_timestamps(self):
        '''
        Generates timestamps based on the sample period (see globals)
        Returns:
            Dictionary of timestamps for all traces, fw traces, and rev traces
        '''
        times = {}
        times['all'] = [np.round((i*SP)/60,2) for i in range(self.num_traces)]
        times['fw'] = times['all'][::2].copy()
        times['rev'] = times['all'][1::2].copy()

        return times

    def generate_trace_dicts(self):
        '''
        Creates dictionaries containing traces according to the experimental timeline
        Returns:
            Trace dictionaries for fw and rev channels
        '''
        fwdict = {'baseline':[], 'TBS':[], '45post-TBS':[], '': []}
        revdict = {'baseline':[], 'TBS':[], '45post-TBS':[], '':[]}
        N = len(self.traces)
        for i in range(N):
            if i%2==0:
                if self.timeline['all'][i] == 'baseline':
                    fwdict['baseline'].append(self.traces[i])
                elif self.timeline['all'][i] == 'TBS':
                    fwdict['TBS'].append(self.traces[i])
                elif self.timeline['all'][i] == '45post-TBS':
                    fwdict['45post-TBS'].append(self.traces[i])
                elif self.timeline['all'][i] == 'end':
                    if 'end' not in fwdict:
                        fwdict['end'] = []
                    fwdict['end'].append(self.traces[i])
                else:
                    fwdict[''].append(self.traces[i])
            if i%2==1:
                if self.timeline['all'][i] == 'baseline':
                    revdict['baseline'].append(self.traces[i])
                elif self.timeline['all'][i] == 'TBS':
                    revdict['TBS'].append(self.traces[i])
                elif self.timeline['all'][i] == '45post-TBS':
                    revdict['45post-TBS'].append(self.traces[i])
                elif self.timeline['all'][i] == 'end':
                    if 'end' not in revdict:
                        revdict['end'] = []
                    revdict['end'].append(self.traces[i])
                else:
                    revdict[''].append(self.traces[i])
        return fwdict, revdict

    def generate_stats(self):
        '''
        Calculates scipy statistical summary for slope, min, and peak.
        Returns:
            Dictionaries containing a simple list of stats for fw and reverse channels, plus a list of the
            corresponding stat labels
        '''
        fwstats = {'slope':{}, 'min':{}, 'peak':{}}
        revstats = {'slope':{}, 'min':{}, 'peak':{}}
        statlabels = ['N', 'min', 'max', 'mean', 'variance', 'skewness']
        for i in self.fwdict:
            if len(self.fwdict[i]) > 1:
                fwstats['slope'][i] = [*sp.stats.describe([j.tf_slope for j in self.fwdict[i]])]
                fwstats['min'][i] = [*sp.stats.describe([j.min[2] for j in self.fwdict[i]])]
                fwstats['peak'][i] = [*sp.stats.describe([j.peak[0] for j in self.fwdict[i]])]
            if len(self.revdict[i]) > 1:
                revstats['slope'][i] = [*sp.stats.describe([j.tf_slope for j in self.revdict[i]])]
                revstats['min'][i] = [*sp.stats.describe([j.min[2] for j in self.revdict[i]])]
                revstats['peak'][i] = [*sp.stats.describe([j.peak[0] for j in self.revdict[i]])]
        return fwstats, revstats, statlabels

    def get_trace(self, num: int):
        '''
        Getter for a specific trace number (should match step #s in .modat file)
        Args:
            num: trace of interest
        Returns:
            Trace object for trace of interest
        '''
        if num > self.num_traces:                   # check if input is valid
            raise Exception(f"Please enter a valid trace number (less than {self.num_traces})")

        num -= 1                                # adjust for zero-based indexing
        ind = num//2                            # take floor of 1/2 trace number

        if num%2 == 0:
            return self.fw[ind]
        else:
            return self.rev[ind]

    def find_tet_time(self):
        '''
        Determines when tetanus occurred based on EPSP minima of fw channel.

        Returns: Tuple
            timestamp -> what time tetanus occurred
            all_ind -> corresponding index is self.traces
            ind -> corresponding index in self.fw
        '''
        arr = [i.min[0] for i in self.fw[:10]]          # fill array with 10 trace baseline from self.fw
        thresh = np.average(arr) - 5*np.std(arr)        # set threshold to 5-sigma deviation from average of arr
        ind = 0                                         # initialize index to 0
        while self.fw[ind].min[0] > thresh:             # increment until threshold is passed
            ind += 1

        all_ind = self.traces.index(self.fw[ind])       # get corresponding self.traces index
        timestamp = self.timestamps['all'][all_ind]     # get corresponding timestamp

        return (timestamp, all_ind, ind)

    def generate_timeline(self):
        '''
        Generates an experimental timeline (Baseline, TBS, 45post) based on the time of tetanization.

        Note:
        - This function will need to be updated if the LTP timeline is changed.
        - The function assumes the 'SP' variable (sampling period) and other necessary variables are defined in the current context.

        Returns:
            timeline (dict): Dictionary of experimental timelines for all traces, fw stims, and rev stims.
            timeline2 (dict): Same as 'timeline', but with duplicated labels removed.
        '''
        # Get time of tetanization and calucate other relevant time points
        tet = self.tet_time[1]                                          # index
        ttime = self.tet_time[0]                                        # time (minutes)
        num = ttime + 45
        tet45 = min(self.timestamps['fw'], key=lambda x: abs(x - num))  # time for 45post
        tet45ind = self.timestamps['all'].index(tet45)                  # index for 45post
        fortyfive = tet45ind - tet                                      # number of indices (traces) to cover 45 minutes
        fifteen = int(np.ceil(15 * 60 / SP))                            # indices for 15 minutes

        timeline = {}
        # Generate timeline for all traces, including 'pre-baseline', 'baseline', 'TBS', and '45post-TBS' phases
        timeline['all'] = ['pre-baseline'] * (tet - int(np.ceil(10 * 60 / SP))) \
                          + ['baseline'] * (int(np.ceil(10 * 60 / SP))) \
                          + ['TBS'] * 6 \
                          + [''] * (fortyfive - 6) \
                          + ['45post-TBS'] * fifteen
        timeline['all'] += ['end'] * (len(self.traces) - len(timeline['all']))

        # Generate separate timelines for forward ('fw') and reverse ('rev') traces
        timeline['fw'] = [i for i in timeline['all'][0::2]]
        timeline['rev'] = [i for i in timeline['all'][1::2]]

        # Create timeline2 by removing duplicated labels
        timeline2 = {}
        for i in timeline:
            timeline2[i] = [''] * len(timeline[i])
            arr = timeline[i]
            junk, inds = np.unique(arr, return_index=True)
            for j in inds:
                timeline2[i][j] = arr[j]

        return timeline, timeline2

    def display_traces(self,
                       traces: list,
                       autotrim=False,
                       measures=False,
                       overlay=False,
                       d=True,
                       color='',
                       label='',
                       slope_overlay=False):
        '''
        Displays graphs of desired traces.

        Args:
            trace_nums (list): List of trace numbers to display.
            autotrim (bool): Whether or not to automatically trim the graph. Default is False.
            measures (bool): Whether to highlight measures (peak, slope, etc.) on the graph. Default is False.
            overlay (bool): Whether to overlay the traces on a single graph or display separate graphs for each trace.
                                Default is False.
            d (bool): Whether to create a new pyplot figure (True) or use the current figure (False). Default is True.
            color (str): Color of the trace(s). Used when overlaying multiple traces on the same graph.
            label (str): Label for the trace when overlaying multiple traces on the same graph.
            slope_overlay (bool): Whether to overlay the best fit line on the trace(s). Default is False.

        Returns:
            None. Displays the pyplot graph(s) of the desired traces.

        Note:
            - The 'trace_nums' parameter should contain integers representing the trace numbers to be displayed.
            - If 'd' is set to False, the traces will be added to the current active figure.
            - If 'overlay' is True, all traces will be displayed on the same graph.
            - If 'overlay' is False, each trace will be displayed on a separate graph or multiple graphs if there are more than six traces.
            - The 'autotrim' option allows trimming of the graph to remove empty spaces around the traces.
            - The 'measures' option adds markers for measures (peak, slope, etc.) on the trace(s) if True.
            - The 'slope_overlay' option overlays slope information on the trace(s) if True.
            - When 'autotrim' is True, the y-axis limits will be automatically adjusted based on the trace measures to improve visibility.
            - If 'color' and 'label' are provided, the traces will be displayed with the specified color and labeled accordingly when using overlay.
            - If 'color' and 'label' are not provided, default colors will be used for each trace.
        '''

        if d:
            plt.figure()

        N = len(traces)

        # if traces[0] is not int:
        #     trace_nums = []
        #     if traces[0][0] is not int:
        #         raise Exception("Please input step numbers as integer tuples (ie. Step 1-20 => (1,20)")
        #     for step in traces:
        #         if step[0] == 1:
        #             trace_nums.append(step[1]*2-1)
        #         elif step[0] == 2:
        #             trace_nums.append(step[1]*2)
        #         else:
        #             raise Exception("Please input step numbers as integer tuples (ie. Step 1-20 => (1,20)")
        # else:
        trace_nums = traces

        if N == 1:
            # Display a single trace
            if slope_overlay:
                self.get_trace(trace_nums[0]).slope_overlay()
            else:
                self.get_trace(trace_nums[0]).show(autotrim=autotrim, measures=measures)
        else:
            if not overlay:
                # Display multiple traces in separate subplots
                trace_arr = [[]]
                c1 = 0
                c2 = 0
                for trace in trace_nums:
                    if c1 <= 6:
                        trace_arr[c2].append(trace)
                        c1 += 1
                    else:
                        trace_arr.append([])
                        c2 += 1
                        c1 = 1
                        trace_arr[c2].append(trace)

                if N == 2:
                    dim = [1, 2]
                elif c2 == 0:
                    dim = [2, c1 % 2 + c1 // 2]
                else:
                    dim = [c2 + 1, 6]

                for i in range(N):
                    plt.subplot(*dim, i + 1)
                    trace = self.get_trace(trace_nums[i])
                    if slope_overlay:
                        trace.slope_overlay(d=False)
                    else:
                        trace.show(False, autotrim=False, measures=measures)

                    if autotrim:
                        if trace_nums[i] % 2 == 1:
                            plt.ylim([self.fwmin[0] - 0.2*SCALE, trace.peak[0] + 0.2*SCALE])
                        else:
                            plt.ylim([self.revmin[0] - 0.2*SCALE, trace.peak[0] + 0.2*SCALE])
                        plt.xlim([0, 30])

            else:
                # Display multiple traces on the same graph
                for i in range(N):
                    trace = self.get_trace(trace_nums[i])
                    if i == 0:
                        if slope_overlay:
                            trace.slope_overlay(d=False)
                        else:
                            trace.show(d=False, measures=measures, color=color, label=label)
                    else:
                        trace.show(d=False, measures=measures, color=color)

                    if autotrim:
                        if trace_nums[i] % 2 == 1:
                            plt.ylim([self.fwmin[0] - 0.2*SCALE, trace.peak[0] + 0.5*SCALE])
                        else:
                            plt.ylim([self.revmin[0] - 0.2*SCALE, trace.peak[0] + 0.5*SCALE])
                        plt.xlim([0, 30])

        if d:
            plt.suptitle(self.ID)
            plt.show()

    def trace_summary(self, ch='fw', autotrim=True, all=False, fw=True):
        '''
        Displays the trace summary overlay for a given channel.

        Args:
            ch (str): Designation of the channel to be displayed.
                - 'fw': Experimental forward channel (default).
                - 'rev': Experimental reverse channel.
                - '1' through '64': Non-experimental channels.
            autotrim (bool): Boolean for whether or not to automatically trim the graph. Default is False.
            all (bool): Boolean for whether to include the full baseline or just the last 10 minutes. Default is False.
            fw (bool): Boolean for whether to use forward stimulation indexing when using a non-experimental channel.
                            Default is True.

        Returns:
            None. Displays the trace summary overlay plot.

        Raises:
            Exception: If the 'ch' argument is invalid (not 'fw', 'rev', or '1~64').

        Note:
            - 'fw': Experimental forward channel, 'rev': Experimental reverse channel.
            - '1' through '64': Non-experiment channels.
            - 'fw' and 'rev' channels are taken from the 'self.fw' and 'self.rev' arrays, respectively.
            - Non-experimental channels are taken from 'self.full' dictionary using the provided 'ch' as the key.
            - The 'autotrim' option allows trimming of the graph to remove empty spaces around the traces.
            - The 'all' option includes the full baseline if True, otherwise only the last 10 minutes are shown.
            - The 'fw' option is relevant when using non-experimental channels and decides whether to use forward
                stimulation indexing (True) or reverse stimulation indexing (False).
        '''

        if ch == 'fw':
            arr = self.fw
            tl = 'fw'
        elif ch == 'rev':
            arr = self.rev
            tl = 'rev'
        else:
            if int(ch) not in range(1, NC + 1):
                raise Exception(f"Please enter a valid channel ('fw', 'rev', or '1~{NC}').")
            elif fw:
                arr = self.full[f'ch{ch}_mV'][::2]
                tl = 'fw'
            else:
                arr = self.full[f'ch{ch}_mV'][1::2]
                tl = 'rev'

        N = len(arr)
        bl = []
        tet = []
        post45 = []

        # Extract indices for 'baseline', 'TBS', and '45post-TBS' events
        for i in range(N):
            if tl == 'fw':
                ind = 1 + i * 2
            else:
                ind = 2 + i * 2
            if self.timeline[tl][i] == 'baseline':
                bl.append(ind)
            elif self.timeline[tl][i] == 'TBS':
                tet.append(ind)
            elif self.timeline[tl][i] == '45post-TBS':
                post45.append(ind)
            elif all:
                bl.append(ind)

        # Plot the trace summary overlay
        plt.figure()

        self.display_traces(tet, autotrim=autotrim, overlay=True, d=False, color=TBScolor, label='TBS')
        self.display_traces(post45, autotrim=autotrim, overlay=True, d=False, color=POST45color, label='45 Post-TBS')
        self.display_traces(bl, autotrim=autotrim, overlay=True, d=False, color=BASELINEcolor, label='Baseline')
        plt.legend(loc='lower right')

        plt.title(f"{self.ID} {ch} Summary")
        plt.show()

    def display_all(self, num, d = True):
        '''
        Displays a pyplot figure of a given trace with subplots showing all 64 channels
        Args:
            num: trace number (NOT STEP NUMBER)
            d: boolean determining whether to display the figure or just initialize it for further modification.
                (pyplot figures are displayed using plt.show())
        Returns:
            None. If d == False, pseudo-returns a pyplot figure as the currently active figure.
        '''
        if num%2 == 1:
            step = f"Step 1-{1+num//2}"
        else:
            step = f"Step 2-{num//2}"
        plt.figure()
        for i in range(64):
            ch = "ch"+str(i+1)+"_mV"
            plt.subplot(8,8,i+1)
            self.full[ch][num-1].show(d=False)
            plt.title(ch)
            plt.ylim([-1.5*SCALE, 1.0*SCALE])

        if d:
            plt.suptitle(f"{self.ID} Trace #{num} ({step})")
            plt.show()

    def plot_measure(self, measure: str, percentage = False, rev = False, d = True):
        if rev:
            ch = self.rev
            timestamps = 'rev'
        else:
            ch = self.fw
            timestamps = 'fw'

        if measure.lower() == 'amplitude':
            arr = [i.amplitude for i in ch]
            label = "EPSP Amplitude "
            units = "(mV)"
            color = MINcolor
        elif measure.lower() == 'min':
            arr = [i.min[0] for i in ch]
            label = "EPSP Amplitude from Baseline "
            units = "(mV)"
            color = MINcolor
        elif measure.lower() == 'peakmin':
            arr = [i.min[3] for i in ch]
            label = "EPSP Amplitude from Peak "
            units = "(mV)"
            color = MINcolor
        elif measure.lower() == 'rawmin':
            arr = [i.min[2] for i in ch]
            label = "EPSP Amplitude "
            units = "(mV)"
            color = MINcolor
        elif measure.lower() == 'fv':
            arr = [i.fiber_volley[0] for i in ch]
            label = "Fiber Volley "
            units = "(mV)"
            color = FVcolor
        elif measure.lower() == 'slope':
            arr = [i.tf_slope for i in ch]
            label = "Slope 10-40% "
            units = "(mV/mS)"
            color = SLOPEcolor
        elif measure.lower() == 'peak':
            arr = [i.peak[0] for i in ch]
            label = "Rebound "
            units = "(mV)"
            color = PEAKcolor
        else:
            raise Exception("Please enter a valid measure.")

        if percentage:
            units = "(% baseline)"
            N = len([i for i in self.timeline[timestamps] if i == 'baseline'])
            avgbl =  np.average(arr[:N])
            arr = [100*(i/avgbl) for i in arr]

        label += units
        if d:
            plt.figure()
        plt.plot(self.timestamps[timestamps], arr, c=color)
        plt.scatter(self.timestamps[timestamps], arr,c=color,s=20)
        plt.xlabel("Time (min)")
        plt.ylabel(label)
        if percentage and measure != 'peak':
            ax = plt.gca()
            ax.invert_yaxis()
        plt.title(label)
        if d:
            plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Trace:
    '''
    Trace Class:
    This is an object for storing each individual trace in an experiment. Each EarlyLTP measure (1040slope, EPSP
    amplitude, etc.) is automatically calculated for each trace upon initialization.
    '''
    def __init__(self, arr: list, num: int, channel: int, ID: str, exp = True,micro=True):
        if micro:
            self.data = [1000*i for i in arr] 
            self.units = "μV"     
        else:
            self.data = arr                                       # raw data for this trace
            self.units = "mV"
        self.mouse = ID
        self.time = \
            [i / 100 for i in range(0, LT * 100, int(ST * 100))]        # time (ms) array for graphing purposes
        self.trace_number = num                                         # corresponding modat trace number
        self.step = (2-num%2, int(np.ceil((num/2))))
        self.step_string = f"Step {self.step[0]}-{self.step[1]}"
        self.ch = channel
        self.baseline = np.round(np.average(arr[:50]), 6)               # average of voltage pre-stimulus
        if exp:
            self.ignore_measures = False
            self.peak = self.find_peak()                                # voltage of post-fiber volley peak
            self.min = self.find_min()                                  # EPSP minimum
            self.thresh = self.find_thresh()
            self.fiber_volley = self.find_fv()                          # fiber volley, measured relative to self.peak
            self.amplitude = self.min[0] - self.peak[0]                 # EPSP amplitude
            self.ten, self.forty = self.find_10_40()                    # 10-40 amplitudes and indices
            self.tf_rise_time = ST*(self.forty[1]                       # 10-40 rise time
                                        - self.ten[1])
            self.tf_linreg = self.calculate_slope()                     # 10-40 slope
            self.tf_slope = self.tf_linreg.slope
            self.tf_rvalue = self.tf_linreg.rvalue
            # self.tf_slope = (self.data[self.forty[1]] -
            #                  self.data[self.ten[1]])/self.tf_rise_time
            self.stimpeak = (max(arr), np.argmax(arr))
        else:
            self.ignore_measures = True


    def __str__(self):
        return f"Trace #: {self.trace_number}\n " \
               f"baseline = {self.baseline},\n " \
               f"fv = {self.fiber_volley[0]}, at {np.round(self.fiber_volley[1]*ST, 2)} ms\n " \
               f"post fv peak = {self.peak[0]}, at {np.round(self.peak[1]*ST, 2)} ms\n " \
               f"trace minimum = {self.min[0]}, at {np.round(self.min[1]*ST, 2)} ms\n " \
               f"fEPSP amplitude = {self.amplitude}\n " \
               f"10% = {self.ten[0]}, at {self.ten[1]*ST} ms\n " \
               f"40% = {self.forty[0]}, at {self.forty[1]*ST} ms\n " \
               f"rise time = {self.tf_rise_time}\n " \
               f"slope = {self.tf_slope}\n"

    def find_peak(self) -> tuple:
        '''
        Extracts the post fiber volley peak
        Returns:
             Tuple -> (magnitude of peak, index in self.data, baseline-adjusted magenitude)
        '''
        ind = int(SA + 0.2//ST)
        while self.data[ind+1] > self.data[ind] or self.data[ind] < self.data[ind-3]:
            ind += 1
        arr = self.data[ind:ind+100].copy()
        peak = np.max(arr) - self.baseline
        
        return (peak, np.argmax(arr)+ind, peak+self.baseline)

    def find_thresh(self):
        arr = self.data[self.peak[1]:self.min[1]]
        return arr.index(min(arr, key=lambda x: abs(x - self.baseline))) + self.peak[1]

    def find_fv(self) -> tuple:
        '''
        Extracts the fiber volley. According to MED64 product documentation, the stimulus artifact last 0.5 ms. Based
        on this fact, and knowing that stim always occurs at 5 ms in our dataset, can extract the fiber volley as the
        local minimum between stimulus and post fv peak
        Returns:
             Tuple -> (magnitude of fiber volley, index in self.data)
        '''
        arr = self.data[SA:self.peak[1]].copy()
        return (np.min(arr)-self.peak[0]-self.baseline, np.argmin(arr)+SA)

    def find_min(self) -> tuple:
        '''
        Extracts the global minimum (not counting the artifact). Same logic as for the fiber volley.
        Returns:
             Tuple -> (magnitude of global minimum, index in self.data, difference from self.peak)
        '''
        min = np.min(self.data[self.peak[1]:].copy())
        ind = np.argmin(self.data[self.peak[1]:].copy())
        return (min-self.baseline, ind+self.peak[1], min, min-self.peak[0])

    def find_10_40(self) -> tuple[tuple[float, int, int], tuple[float, int, int]]:
        '''
        Calculates and returns the amplitudes at 10% and 40% of the time interval from peak to EPSP Minimum, along
        with their corresponding indices.

        Returns:
            Tuple of Tuples, (magnitude, index, timepoint), for 10% and 40% amplitude
        '''

        peak = self.peak[0]
        pind = self.peak[1]
        amp = self.min[0]
        mind = self.min[1]
        arr = self.data[pind:mind]

        # interval = mind - pind
        # ten = int(np.round(pind + 0.1*interval))
        # forty = int(np.round(pind + 0.4*interval))

        ten_percent = amp * 0.1
        forty_percent = amp * 0.4

        ten = arr.index(min(arr, key=lambda x: abs(x - ten_percent)))+pind
        forty = arr.index(min(arr, key=lambda x: abs(x - forty_percent)))+pind

        return ((self.data[ten], ten, ten*ST), (self.data[forty], forty, forty*ST))

    def calculate_slope(self):
        ten = self.ten[1]
        forty = self.forty[1]
        linreg = sp.stats.linregress(self.time[ten:forty+1], self.data[ten:forty+1])
        return linreg

    # def plot(self,time,data,c,label, paper=False):

    def show(self, d = True,
             t1 = 0, t2 = LT,
             in_ms = True,
             autotrim = False,
             measures = False,
             slope = False,
             color = BASELINEcolor, label = '', paper=False):
        '''
        Method for displaying the given trace. All args have default values which will result in displaying the
        entire trace. Use the arguments to delay use of plt.show(), and/or to only graph a specific time interval.
        Args:
            d: Boolean for displaying the trace. Defaults to true, set to false if using plt.subplot(), or if
            you want to make changes to the plot

            t1 and t2: Time interval you would like to display (shows entire trace by default). Units must be in either
            milliseconds or sample number (MUST SET IN_MS TO FALSE IF USING THE LATTER!).

            in_ms: Boolean for whether the time interval units are in milliseconds (defaults to true).

            autotrim: Boolean for automatically setting t1 & t2 to the interval from stim artifact to 20 ms after

            measures: whether or not to highlight measures (default is False)

            slope: whether or not to highlight just the slope, trumped by measures (default is False)

        Returns:
            Technically returns nothing, but the plt object is carried through the method, and can be altered
            further after it has been called.
        '''
        if d:
            plt.figure()

        if autotrim:                                            # autotrim if desired
            t1 = 0
            t2 = int(30/ST)
        elif in_ms:                                             # convert to sample #'s if needed
            t1 = int(t1/ST)
            t2 = int(t2/ST)
        time = self.time
        data = self.data

        if measures and not self.ignore_measures:
            fv = self.fiber_volley[1]
            peak = self.peak[1]
            min = self.min[1]
            ten = self.ten[1]
            ten_val = data[ten]
            forty = self.forty[1]

            plt.plot(time[t1:fv-1], data[t1:fv-1],
                     BASELINEcolor, lw=3, label = "Base Trace")
            plt.plot(time[fv-2:fv+3], data[fv-2:fv+3],
                     FVcolor, lw=3, label = "Fiber Volley")



            plt.plot(time[fv+2:peak-1], data[fv+2:peak-1], BASELINEcolor, lw=3)
            plt.plot(time[peak-2:peak+3], data[peak-2:peak+3],
                     PEAKcolor, lw=3, label = "Rebound")



            plt.plot(time[peak+2:ten+1], data[peak+2:ten+1],BASELINEcolor, lw=3)


            plt.plot(time[ten:forty+1], data[ten:forty+1],
                     SLOPEcolor, lw=3, label = "10-40%")


            plt.text(time[ten]+0.05, ten_val,
                     f"  10-40% Slope = {np.round(self.tf_slope,3)}",
                     ha='left')

            plt.plot(time[forty:min-7], data[forty:min-7], BASELINEcolor, lw=3)
            plt.plot(time[min-8:min+11], data[min-8:min+11],
                     MINcolor, lw=3, label = "EPSP Minimum")

            plt.scatter(time[fv], data[fv], s=40, c=FVcolor, marker=6) # type: ignore
            plt.scatter(time[peak], data[peak], s=40, c=PEAKcolor, marker=7) # type: ignore
            plt.scatter(time[min], data[min], s=40, c=MINcolor, marker = 6) # type: ignore
            plt.scatter(time[ten], data[ten], s=40, c=SLOPEcolor, lw=3, marker=4) # type: ignore
            plt.scatter(time[forty], data[forty], s=40, c=SLOPEcolor, marker=4) # type: ignore

            plt.legend(loc='lower right')

            plt.plot(time[min+10:t2], data[min+10:t2],BASELINEcolor, lw=3)

        elif slope and not self.ignore_measures:
            ten = self.ten[1]
            ten_val = data[ten]
            forty = self.forty[1]
            plt.plot(time[t1:ten+1], data[t1:ten+1],
                     BASELINEcolor, lw=3, label="Base Trace")
            plt.plot(time[ten:forty + 1], data[ten:forty + 1],
                     SLOPEcolor, lw=3, label="10-40%")

            # plt.text(time[ten] + 0.05, ten_val,
            #          f"  10-40% Slope = {np.round(self.tf_slope, 3)}",
            #          ha='left')
            plt.plot(time[forty:t2], data[forty:t2], BASELINEcolor, lw=3)
            plt.scatter(time[ten], data[ten], s=40, c=SLOPEcolor, lw=3, marker=4) # type: ignore
            plt.scatter(time[forty], data[forty], s=40, c=SLOPEcolor, marker=4) # type: ignore
            plt.legend(loc='lower right')

        
        if label != '':
            plt.plot(time[t1:t2 + 1], data[t1:t2 + 1], c=color, label = label)
        else:
            plt.plot(time[t1:t2 + 1], data[t1:t2 + 1], c = color)   # create pyplot object

        if autotrim and d:
            plt.ylim([self.min[2]-0.2*SCALE,
                      self.peak[0]+0.2*SCALE])

        plt.xlabel("Time (ms)")                                            # set labels and title
        plt.ylabel(f"Voltage ({self.units})")
        plt.title(f"Trace #{self.trace_number} ({self.step_string})")
        plt.suptitle(self.mouse)

        # if paper:
        #     fig, axis = plt.subplots()
        #     plt.axis('off')

        #     if d:
        #         plt.show()
        #     else:
        #         return fig,axis
        if d: plt.show()                                            # show if desired
       


    def slope_overlay(self, other = 0, d=True, measures = False):
        if d:
            plt.figure()
        if measures:
            self.show(autotrim=True, measures=True,d=False)
        else:
            self.show(autotrim=True, slope = True, d = False)
        t1 = self.ten[1]
        t2 = self.forty[1]
        arr = self.time[t1-3:self.min[1]-3]
        yvals = [self.tf_linreg.intercept + i*self.tf_slope for i in arr]
        plt.plot(arr, yvals, '--', c = MINcolor, lw = 3)
        if other != 0:
            mean = np.average([self.ten[0], self.forty[0]])
            tmean = np.average([self.time[self.ten[1]], self.time[self.forty[1]]])
            intercept = mean - other*tmean
            yvals = [intercept + i*other for i in arr]
            plt.plot(arr, yvals, '--', c=FVcolor)
            plt.text(self.time[t2] + 0.05, self.data[t2],
                     f"  Mobius 1040 = {np.round(other, 3)}",
                     ha='left')
        plt.ylim([self.min[2] - 0.2*SCALE,
                  self.peak[0] + 0.2*SCALE])
        if d:
            plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ File Handling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_file(ID: str, 
              filepath: str, 
              fw: tuple[int,int], 
              rev: tuple[int,int], 
              hemi: str,
              strengths: tuple[int, int], 
              blind = False) -> LTP:
    '''
    Extracts relevant channels from raw data file and inserts into LTP object
    Args:
        ID: mouse ID
        filepath: filepath for raw data file, file name must be of form "(MOUSE ID)_(Hemisphere/Slice#).csv"
        fw: integer tuple for forward traces -> (stim channel #, response channel #)
        rev: integer tuple for reverse traces
        strengths: Tuple -> (fw stim strength, rev stim strength)

    Returns:
        LTP object containing all of the data from the raw data csv
    '''
    f = open(filepath, 'r')
    if blind:
        fw = (1,1)
        rev = (1,1)
        strengths = (1,1)
        data = LTP(ID, filepath, pd.read_csv(f, skiprows=3), fw, rev, hemi, strengths, blind = True) # type: ignore
        f.close()
        return data
    data = LTP(ID, filepath, pd.read_csv(f, skiprows=3), fw, rev, hemi, strengths) # type: ignore
    f.close()
    return data

def load_pickle(ID: str, 
                filepath: str, 
                fw: tuple[int,int], 
                rev: tuple[int,int], 
                hemi: str,
                strengths: tuple[int, int], 
                blind = False) -> LTP:
    '''
    Extracts relevant channels from raw data file and inserts into LTP object
    Args:
        ID: mouse ID
        filepath: filepath for pickled dataframe, file name must be of form "MOUSEID.pkl"
        fw: integer tuple for forward traces -> (stim channel #, response channel #)
        rev: integer tuple for reverse traces
        strengths: Tuple -> (fw stim strength, rev stim strength)

    Returns:
        LTP object containing all of the data from the raw data csv
    '''
    f = filepath
    if blind:
        fw = (1,1)
        rev = (1,1)
        strengths = (1,1)
        data = LTP(ID, filepath, pd.read_pickle(f), fw, rev, hemi, strengths, blind = True) # type: ignore
        
        return data
    data = LTP(ID, filepath, pd.read_pickle(f), fw, rev, hemi, strengths) # type: ignore
    return data

def batch_from_csv(metapath: str, 
                   subID: tuple[str, int, int], 
                   mset: list = [], 
                   datapath=os.getcwd(),
                   pickle_it = False) -> dict:
    '''
    Function for running batches of data. Processes and loads data from .csv files into LTP objects for each mouse.

    Args:
        metapath (str): Filepath to the CSV file with stim and response info for each mouse.
        subID (tuple): A tuple containing:
                        - identifying substring from the mouse ID (e.g., "22" for the 2022 summer pilot).
                        - length of full mouse ID strings (e.g., 4 for the 2022 summer pilot).
                        - length of the hemisphere substring (e.g., 3 if "RH1", 4 if "RHS1").
        mset (list): IDs of mice to be analyzed. Defaults to an empty list, in which case the program will assume you
                        want to analyze ALL mice in the directory.
        datapath (str): Filepath to the directory containing the raw data .csv's. Defaults to the current working directory.

    Returns:
        dict: A dictionary of LTP objects containing data for each mouse.

    NOTE: 'mset' defaults to empty, meaning the function will analyze ALL mice in the directory if 'mset' is not specified.
    '''

    # Extract information about the data files based on the metadata CSV file
    IDsubstring = subID[0]
    IDlengths = [subID[1]]
    for i in subID[2:]:
        IDlengths.append(subID[1] + 1 + i)

    animal_IDs, hemis, filenames, fw_stims, fw_responses, \
    rev_stims, rev_responses, strengths = extract_metadata(metapath, IDsubstring, IDlengths) # type: ignore

    N = len(animal_IDs)

    # Create a list of raw IDs with hemisphere/slice info for matching with 'mset'
    raw_set = [''] * N
    for i in range(N):
        raw_set[i] = f"{animal_IDs[i]}_{hemis[i]}"

    mset_size = len(mset)
    using_hemis = [(len(i) > IDlengths[0]) for i in mset]

    # Find indices of the mice in 'mset' for further processing
    if mset_size > 0:
        indices = [0] * mset_size
        for i in range(mset_size):
            if using_hemis[i]:
                indices[i] = raw_set.index(mset[i])   # For 'mset[i]' having ID + hemi/slice
            else:
                indices[i] = animal_IDs.index(mset[i])  # For 'mset[i]' only having ID
    else:
        indices = [i for i in range(N)]  # Use full directory if 'mset' is not specified

    # Create a list of directory paths to locate filepaths for mice of interest
    dir = []
    for root, directories, files in os.walk(datapath):
        for name in files:
            if re.search('raw.*csv', name.lower()):
                dir.append(os.path.join(root, name))

    not_exist = []
    datafiles = {}

    # Locate filepaths for mice of interest
    for i in indices:
        file = filenames[i]
        file_exists = False
        for j in dir:
            if file.lower() in j.lower():
                # Search for raw data .csv file for the current mouse
                if using_hemis[indices.index(i)]:
                    datafiles[raw_set[i]] = j
                else:
                    datafiles[animal_IDs[i]] = j
                file_exists = True
                break

        if not file_exists:
            not_exist.append(file)  # Keep track of non-existent files for error handling

    # Raise an exception if some datafiles couldn't be found
    if len(not_exist) != 0:
        raise Exception(f"Could not detect the following datafiles: {not_exist[:]}.\n"
                        f"Please ensure there is a .csv raw data file for each mouse in the directory.\n"
                        f".xlsx doesn't work!!")

    # Generate a dictionary of LTP data for each mouse
    batch = {}
    for i in indices:
        if using_hemis[indices.index(i)]:
            ID = raw_set[i]
        else:
            ID = animal_IDs[i]

        # Load LTP object for each mouse into the batch
        batch[ID] = load_file(ID, datafiles[ID],
                              (fw_stims[i], fw_responses[i]),
                              (rev_stims[i], rev_responses[i]),
                              hemis[i],
                              strengths[i])
        if pickle_it:
            batch[ID].pickle_df(datapath)
    return batch



def batch_from_pkl(metapath: str, 
                   subID: tuple[str, int, int], 
                   mset: list, 
                   datapath = os.getcwd()) -> dict:
    '''
    Function for running batches of data. Should be able to locate the files as long as they are all somewhere in the
    datapath directory. Probably a good idea to narrow down the search at least a bit so it's not searching through
    thousands or millions of files.
    Args:
        datapath: filepath to directory containing the dataframe .pkl's, defaults to current working directory
        metapath: filepath to csv file with stim and response info for each mouse
        subID: tuple -> (identifying substring from mouse ID (ie. "22" for the 2022 summer pilot),
                            length of full mouse ID strings (ie. 4 for the 2022 summer pilor),
                            length of hemisphere substring (ie. 3 if "RH1", 4 if "RHS1")
        mset: list -> IDs of mice to be analyzed
    NOTE: mset defaults to empty, in which case the program will assume you want to analyze ALL mice in the directory!!

    Returns:
        dictionary of LTP objects containing data for each mouse
    '''

    IDsubstring = subID[0]
    IDlengths = [subID[1]]
    for i in subID[2:]:
        IDlengths.append(subID[1]+1+i)

    # extract metadata
    animal_IDs, hemis, filenames, fw_stims, fw_responses, rev_stims, \
    rev_responses, strengths = extract_metadata(metapath, IDsubstring, IDlengths)   # type: ignore
    N = len(animal_IDs)

    raw_set = ['']*N                                                # put together animal IDs and hemis to match mset
    for i in range(N):                                              #    string format and metadata indices
        raw_set[i] = f"{animal_IDs[i]}_{hemis[i]}"

    mset_size = len(mset)
    if mset_size > 0:
        using_hemis = [(len(i) > IDlengths[0]) for i in mset]
    else:
        using_hemis = [False for i in range(N)]

    if mset_size > 0:
        indices = [0]*mset_size
        for i in range(mset_size):                                      # find indices of the mice in mset
            if using_hemis[i]:
                ind = raw_set.index(mset[i])
                indices[i] =  ind                                       # for mset[i] having ID + hemi/slice
                filenames[ind] = f'{raw_set[ind]}_df.pkl'
            else:
                ind = animal_IDs.index(mset[i])                         # for mset[i] only having ID
                indices[i] = ind
                filenames[ind] = f'{animal_IDs[ind]}_df.pkl'
    else:
        indices = [i for i in range(N)]                                 # uses full directory if mset is not specified
        for i in range(N):
            filenames[i] = f'{animal_IDs[i]}_df.pkl'
        for i in range(N):
            if filenames.count(filenames[i]) > 1:
                filenames[i] = f'{raw_set[i]}_df.pkl'
                using_hemis[i] = True
    dir = []
    for root, directories, files in os.walk(datapath):
        for name in files:
            if re.search('df.*.pkl', name.lower()):
                dir.append(os.path.join(root,name))

    not_exist = []
    datafiles = {}

# locate filepaths for mice of interest

    for i in indices:                                       # iterate through indices for mice in mset
        file = filenames[i]
        file_exists = False
        for j in dir:                                       # iterate through file directory
            if file.lower() in j.lower():                   # search for raw data .csv file for current mouse
                if using_hemis[indices.index(i)]:
                    datafiles[raw_set[i]] = j               # add full filepath to datafiles
                else:
                    datafiles[animal_IDs[i]] = j
                file_exists = True                          # flag that the file does exist
                break
        if not file_exists:
            not_exist.append(file)                          # error handling for nonexistent file

    if len(not_exist) != 0:
        raise Exception(f"Could not detect the following datafiles: {not_exist[:]}.\n"
                        f"Please ensure there is a .pkl file for each mouse in the directory!!")

# generate dictionary of EarlyLTP data
    batch = {}                                                      # dictionary for batch data
    for i in indices:
        if using_hemis[indices.index(i)]:
            ID = raw_set[i]
        else:
            ID = animal_IDs[i]
        batch[ID] = load_pickle(ID, datafiles[ID],                    # load LTP object for each mouse into the batch
                              (fw_stims[i], fw_responses[i]),
                              (rev_stims[i],rev_responses[i]),
                              hemis[i],
                              strengths[i])
    return batch

def batch_from_any(metapath: str, 
                   subID: tuple[str, int, int], 
                   mset: list = [], 
                   datapath=os.getcwd(),
                   pickle_it = False) -> dict:
    
    '''
    Same as batch_from_csv() and batch_from_pkl(), but will search for both file types
    and only load in from a csv if a pickly is unavailable. 

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    '''
    IDsubstring = subID[0]
    IDlengths = [subID[1]]
    for i in subID[2:]:
        IDlengths.append(subID[1] + 1 + i)

    animal_IDs, hemis, filenames, fw_stims, fw_responses, \
    rev_stims, rev_responses, strengths = extract_metadata(metapath, IDsubstring, IDlengths) # type: ignore

    N = len(animal_IDs)

    # Create a list of raw IDs with hemisphere/slice info for matching with 'mset'
    raw_set = [''] * N
    for i in range(N):
        raw_set[i] = f"{animal_IDs[i]}_{hemis[i]}"

    mset_size = len(mset)
    using_hemis = [(len(i) > IDlengths[0]) for i in mset]

    # Find indices of the mice in 'mset' for further processing
    if mset_size > 0:
        indices = [0] * mset_size
        for i in range(mset_size):
            if using_hemis[i]:
                indices[i] = raw_set.index(mset[i])   # For 'mset[i]' having ID + hemi/slice
            else:
                indices[i] = animal_IDs.index(mset[i])  # For 'mset[i]' only having ID
    else:
        indices = [i for i in range(N)]  # Use full directory if 'mset' is not specified

    # Create a list of directory paths to locate filepaths for mice of interest
    dir = []
    pkls = []
    for root, directories, files in os.walk(datapath):
        for name in files:
            if re.search('raw.*csv', name.lower()):
                dir.append(os.path.join(root, name))
            elif re.search('df.*pkl', name.lower()):
                pkls.append(os.path.join(root, name))

    not_exist = []
    datafiles = {}

    pklnames = []
    for f in filenames:
        fname_init = f.split('.')[0].split('_')
        fname = ''
        for i in fname_init:
            if i.lower() != 'raw':
                fname += i+'_'
        fname += "df.pkl"
        pklnames.append(fname)

    # Locate filepaths for mice of interest
    for i in indices:
        file_exists = False

        # Search for pickles first
        for j in pkls:
            file = pklnames[i]
            if file.lower() in j.lower():
                # Search for pickled dataframe for the current mouse
                if using_hemis[indices.index(i)]:
                    datafiles[raw_set[i]] = j
                else:
                    datafiles[animal_IDs[i]] = j
                file_exists = True
                break

        # Search for csv if no pickle found
        if not file_exists:
            for j in dir:
                file = filenames[i]
                if file.lower() in j.lower():
                    # Search for raw data .csv file for the current mouse
                    if using_hemis[indices.index(i)]:
                        datafiles[raw_set[i]] = j
                    else:
                        datafiles[animal_IDs[i]] = j
                    file_exists = True
                    break

        if not file_exists:
            not_exist.append(file)  # Keep track of non-existent files for error handling

    # Raise an exception if some datafiles couldn't be found
    if len(not_exist) != 0:
        raise Exception(f"Could not detect the following datafiles: {not_exist[:]}.\n"
                        f"Please ensure there is a .csv raw data file for each mouse in the directory.\n"
                        f".xlsx doesn't work!!")

    # Generate a dictionary of LTP data for each mouse
    batch = {}
    for i in indices:
        if using_hemis[indices.index(i)]:
            ID = raw_set[i]
        else:
            ID = animal_IDs[i]

        if 'pkl' in datafiles[ID]:
            # Load LTP object for each mouse into the batch
            batch[ID] = load_pickle(ID, datafiles[ID],
                                (fw_stims[i], fw_responses[i]),
                                (rev_stims[i], rev_responses[i]),
                                hemis[i],
                                strengths[i])
        else:

            # Load LTP object for each mouse into the batch
            batch[ID] = load_file(ID, datafiles[ID],
                                (fw_stims[i], fw_responses[i]),
                                (rev_stims[i], rev_responses[i]),
                                hemis[i],
                                strengths[i])
            if pickle_it:
                batch[ID].pickle_df(datapath)

    return batch

def extract_metadata(metapath: str, IDsubstring: str, IDlengths: tuple) -> tuple:
    '''
    Extracts metadata from the directory csv file. Formatting is quite flexible, only requirements are that data is
    arranged in columns, with the mouse IDs (ie. B22R, Y22L, etc.) acting as a header, with labels in the first cell of
    each row. Data labels are fairly flexible as well, and should work with most informative labelling methods. IT IS
    IMPORTANT TO NOTE, HOWEVER, THAT STIM AND RESPONSE CHANNELS MUST BE IN SEPARATE ROWS, EACH WITH THEIR OWN LABEL!!
    When in doubt, write the label in plain english (ie. "Forward Stimulus Channels", etc.). THE ANIMAL ID ROW LABEL
    IS NOT FLEXIBLE!! It must contain the term "ID" in UPPER CASE somewhere within the cell.
    Args:
        metapath: filepath for directory csv file
        IDsubstring: same as subID[0] from run_batch()
        IDlengths: same as subID[1:] from run_batch()

    Returns:
        list of metadata extracted from the directory.
    '''
    if '.csv' not in metapath:                                              # error handling for non-csv
        raise Exception("Please ensure your directory is a .csv (UTF-8) file!")

    f = open(metapath, 'r')                                                 # open metadata file

    l = f.readline().rstrip().split(',')                                             # load first line into an array

    while "ID" not in l[0]:                                                 # iterate to row with mouse IDs
        l = f.readline().rstrip().split(',')

    animal_IDs = [i for i in l if IDsubstring in i and len(i) in IDlengths] # load mouse IDs
    N = len(animal_IDs)

    tracker = {}                                                            # initialize tracker dictionary
    keys = ['fw_stims', 'fw_responses', 'rev_stims', 'hemisphere/slice',
               'filenames', 'rev_responses', 'fw_strengths', 'rev_strengths']
    for i in keys:
        tracker[i] = False                                                  # initialize all to False

    while l[0] != '':                                                       # extract relevant info
        l = f.readline().rstrip().split(',')
        if re.search("hemi" or "slice", l[0].lower()):
            hemis = [i for i in l[1:N+1]]                                   # hemisphere/slice
            tracker['hemisphere/slice'] = True
        if re.search("file", l[0].lower()):                                 # names of datafiles
            filenames = [i for i in l[1:N+1]]
            tracker['filenames'] = True
        if re.search("f.*w.*stim.*ch", l[0].lower()):
            fw_stims = [int(i) for i in l[1:N+1]]                           # forward stim channels
            tracker['fw_stims'] = True                                      # set tracker to True if data is found
        if re.search("f.*w.*res", l[0].lower()):
            fw_responses = [int(i) for i in l[1:N+1]]                       # forward response channels
            tracker['fw_responses'] = True
        if re.search("f.*w.*stim.*str", l[0].lower()):                      # forward stim strengths
            fw_str = [i for i in l[1:N+1]]
            tracker['fw_strengths'] = True
        if re.search("rev.*stim.*ch", l[0].lower()):
            rev_stims = [int(i) for i in l[1:N+1]]                          # reverse stim channels
            tracker['rev_stims'] = True
        if re.search("rev.*res", l[0].lower()):
            rev_responses = [int(i) for i in l[1:N+1]]                      # reverse response channels
            tracker['rev_responses'] = True
        if re.search("rev.*stim.*str", l[0].lower()):
            rev_str = [int(i) for i in l[1:N+1]]                            # reverse stim strengths (in uA)
            tracker['rev_strengths'] = True
    f.close()

    if not tracker['fw_strengths'] or not tracker['rev_strengths']:         # stim strength is nonessential
        strengths = [(None,None)]*N
        tracker['fw_strengths'] = True
        tracker['rev_strengths'] = True
    else:
        strengths = [(fw_str[i], rev_str[i]) for i in range(N)]

    errors = [i for i in keys if not tracker[i]]

    if len(errors) != 0:                                                    # error handling for any absent data
        raise Exception(f"The following vital information is absent from the metadata file: \n"
                        f"{errors}")

    return (animal_IDs, hemis, filenames, fw_stims, fw_responses, rev_stims, rev_responses, strengths)


def batch_to_csv(batch: dict, micro=True, filepath='', subfolder='', append='', order=list, overwrite=False):
    '''
    Takes a dictionary of batch data and exports the measures to a .csv

    Args:
        batch (dict): A dictionary of LTP objects returned by run_batch().
        micro (bool): Boolean for whether to output units as micro (µV) or milli (mV) units.
                      Defaults to True, so it matches the .modat file units.
        filepath (str): Destination for csv files. Defaults to an empty string, and files will be written to the
                            current working directory (cwd).
        subfolder (str): Name for a subfolder to create for batch csv's. Defaults to none, and all csv's are dumped
                            into 'Auto_Measures'.
        append (str): If a filename is provided, the data will be appended to an existing .csv file with the given name.
        order (list): A list containing the mouse IDs in the desired order of appearance in the final .csv file.
                      Defaults to an empty list, and the data is written in the order they appear in the 'batch' dictionary.
        overwrite (bool): If True, allows overwriting of existing files. Defaults to False, where an exception will
                            be raised if the file already exists.
    Returns:
        None
    '''

    # Initialize empty dictionaries to store DataFrames and channels for each mouse
    dfs = {}
    chs = {}

    # Loop through each mouse in the batch
    for mouse in batch:

        # Get data for the current mouse
        data = batch[mouse]
        dfs[mouse + '_fw'], dfs[mouse + '_rev'] = [], []

        # Create strings for channel information (Stim, Response)
        fwchs = f"Stim, Response: ch{data.fw_s}, ch{data.fw_r}"
        revchs = f"Stim, Response: ch{data.rev_s}, ch{data.rev_r}"

        # Get the number of forward and reverse traces
        numfw = len(data.fw)
        numrev = len(data.rev)

        # Extract Trace Measures
        if micro:
            u = "µ"  # Micro symbol for unit display
            measures_fw = [(1000 * i.min[2], 1000 * i.tf_slope, i.tf_rvalue, 1000 * i.fiber_volley[0]) for i in data.fw]
            measures_rev = [(1000 * i.min[2], 1000 * i.tf_slope, i.tf_rvalue, 1000 * i.fiber_volley[0]) for i in data.rev]
        else:
            u = "m"  # Milli symbol for unit display
            measures_fw = [(i.min[2], i.tf_slope, i.tf_rvalue, i.fiber_volley[0]) for i in data.fw]
            measures_rev = [(i.min[2], i.tf_slope, i.tf_rvalue, i.fiber_volley[0]) for i in data.rev]

        # Set up time column and row labels for forward and reverse traces
        fw_time = data.timestamps['fw']
        rev_time = data.timestamps['rev']
        fw_rows = [f"Step1-{i+1}" for i in range(numfw)]
        rev_rows = [f"Step2-{i+1}" for i in range(numrev)]

        # Generate a list of tuples for each row containing trace measures
        fw_tuples = [(fw_rows[i], data.timeline['fw'][i], fw_time[i], *measures_fw[i]) for i in range(numfw)]
        rev_tuples = [(rev_rows[i], data.timeline['rev'][i], rev_time[i], *measures_rev[i]) for i in range(numrev)]

        # Define column names for the DataFrame
        column_names = ['Trace#', 'Phase', 'Time (min)', f'EPSP Amplitude ({u}V)', f'Slope1040 ({u}V/ms)', f'1040 R-value', f'Fiber Volley ({u}V)']

        # Append mouse ID to each row if 'append' parameter is provided
        if append != '':
            fw_tuples = [(data.ID+'_fw', *i) for i in fw_tuples]
            rev_tuples = [(data.ID+'_rev', *i) for i in rev_tuples]
            column_names = ['ID'] + column_names

        # Create pandas DataFrame for forward and reverse traces, and store them in the dictionaries
        dfs[mouse + '_rev'] = pd.DataFrame(rev_tuples, columns=column_names)
        chs[mouse + '_rev'] = revchs
        dfs[mouse + '_fw'] = pd.DataFrame(fw_tuples, columns=column_names)
        chs[mouse + '_fw'] = fwchs

    # Go to specified destination if needed
    if filepath != '':
        current_directory = os.getcwd()
        os.chdir(filepath)

    # Make a new folder and write each DataFrame into a .csv file
    if append != '':
        with open(f"{append}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(column_names)
            f.close()

        # Append the data to the .csv file based on the order provided, if any
        if len(order) == 0: # type: ignore
            for mouse in dfs:
                dfs[mouse].to_csv(f"{append}.csv", index=False, encoding="utf-8-sig", mode='a', header=False)
        else:
            for mouse in order: # type: ignore
                dfs[mouse].to_csv(f"{append}.csv", index=False, encoding="utf-8-sig", mode='a', header=False)

    else:
        # Create a new folder for storing batch data if 'subfolder' parameter is provided
        os.makedirs(os.path.join('Auto_Measures',subfolder), exist_ok=True)

        # Write each DataFrame into separate .csv files for each mouse
        for mouse in dfs:
            filename = f"{mouse}_Measures.csv"
            path = os.path.join('Auto_Measures',subfolder,filename)

            # Raise an exception if the file already exists and overwrite is disabled
            if os.path.exists(path) and not overwrite:
                raise Exception(f"A measures.csv file already exists for {mouse}, please choose a different destination, or enable overwrite!")

            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f"{mouse} Auto Measures", f"{chs[mouse]}"])
                f.close()
            dfs[mouse].to_csv(path, index=False, encoding="utf-8-sig", mode='a')

    # Return to the original working directory if needed
    if filepath != '':
        os.chdir(current_directory)

def stats_to_csv(batch: dict, percentage=False, append='', subfolder='', order=list, filepath='', overwrite=True):
    '''
    Exports the scipy statistical summary for the measures of each mouse in a batch

    Args:
        batch (dict): A dictionary containing the LTP class objects for each mouse.
        percentage (bool): Determines whether to convert to percentages. Default is False.
        append (str): Determines whether to create and append the stats of each mouse to a single .csv,
                      or if there should be a separate .csv for each mouse.
                      - DEFAULT -> Empty string ('') which equates to creating a separate .csv for each mouse.
                      - USAGE -> A string containing the desired filename (DO NOT INCLUDE '.csv').
        subfolder (str): Determines whether or not to create a subfolder to hold the new .csv's.
                         - DEFAULT -> Empty string, which means it will dump the .csv's directly into the destination folder.
                         - USAGE -> A string containing the desired foldername.
        order (list): If you are appending the stats to a single .csv, this variable allows you to determine the order in
                      which they are appended. Should be a list of strings matching the ID string for each mouse.
        filepath (str): This is a string containing the filepath for the directory where you want the .csv's.
                        - DEFAULT -> Empty string, which means it will write to the current working directory.
                        - USAGE -> A string containing the desired filepath (MAKE SURE TO FORMAT PROPERLY, '/' for UNIX or macOS
                                   and '\\' for windows. 
                                   FOR WINDOWS:
                                    Use f"filepath" with double backslash characters ('\\') to ensure Python properly
                                    interprets the backslash ('\') character. NOTE: using r"filepath" with single backslash
                                    characters also works, but you CANNOT have a single backslash at the end of a string!!!
                                    you'll have to do a double backslash at the end and remove the extra one afterwards).
        overwrite (bool): A boolean variable for whether or not to overwrite existing files.
                          - DEFAULT -> True, which means it will overwrite any existing files should a filename conflict arise.
                          - USAGE -> False, which means it will raise an exception should a filename conflict arise.

    Returns:
        None. Creates .csv(s) containing the measure summary statistics for each mouse in the batch.
    '''

    statarrs = {}
    key = ''
    for mouse in batch:
        # Get stats for each mouse in the batch
        data = batch[mouse]
        fwarr, revarr = [], []

        # Extract statistical measures for each phase ('baseline', 'TBS', '45post-TBS')
        for phase in ['baseline', 'TBS', '45post-TBS']:
            tempfw, temprev = [], []
            for i in ['slope', 'min', 'peak']:
                tempfw += [j for j in data.fwstats[i][phase]]
                temprev += [j for j in data.revstats[i][phase]]
            fwarr.append([phase] + tempfw)
            revarr.append([phase] + temprev)
        statarrs[mouse + '_fw'] = fwarr
        statarrs[mouse + '_rev'] = revarr

        if key == '':
            key = mouse

    # Define column names for the DataFrame
    column_names = ['Phase'] + batch[key].statlabels
    N = len(column_names) - 2
    supcolumns = [''] + ['Slope1040 (mV/ms)'] + [''] * N + ['EPSP Amplitude (mV)'] + [''] * N + ['Rebound (mV)']

    # Go to the specified destination if needed
    if filepath != '':
        current_directory = os.getcwd()
        os.chdir(filepath)

    # Make a new folder and write each DataFrame into a .csv file
    if append != '':
        column_names = ['ID'] + ['Phase'] + column_names[1:] * 3
        supcolumns = [''] + supcolumns

        with open(f"{append}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(supcolumns)
            writer.writerow(column_names)

            # Append the stats to the .csv file based on the order provided, if any
            if len(order) == 0: # type: ignore
                for mouse in statarrs:
                    for i in statarrs[mouse]:
                        writer.writerow([mouse] + i)
            else:
                for mouse in order: # type: ignore
                    for i in statarrs[mouse]:
                        writer.writerow([mouse] + i)
            f.close()
    else:
        os.makedirs(os.path.join('PyStats',subfolder), exist_ok=True)

        # Write each DataFrame into separate .csv files for each mouse
        for mouse in statarrs:
            filename = f"{mouse}_Stats.csv"
            path = os.path.join('PyStats',subfolder,filename)

            # Raise an exception if the file already exists and overwrite is disabled
            if os.path.exists(path) and not overwrite:
                raise Exception(f"A measures.csv file already exists for {mouse}, please choose a different destination, or enable overwrite!")

            with open(f"{append}.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(supcolumns)
                writer.writerow(column_names)

                # Write the stats for each phase to the .csv file
                for i in statarrs[mouse]:
                    writer.writerow(i)
            f.close()

    # Return to the original working directory if needed
    if filepath != '':
        os.chdir(current_directory)

def display_measures(batch: dict, percentage=False):
    '''
    Creates and displays a pyplot figure showing the measures for each mouse in a batch.

    Args:
        batch (dict): A dictionary containing the LTP class objects for each mouse.
        percentage (bool): Boolean variable for whether to display measures using percentage or normal units.
                           - If True, measures will be displayed using percentage units.
                           - If False, measures will be displayed using normal units.
    '''

    # Loop through each mouse in the batch and display the measures using subplots
    for i in batch:
        mouse = batch[i]

        # Create a new pyplot figure
        plt.figure()

        # Plot the 'peak' measure in the first subplot
        plt.subplot(3, 1, 1)
        mouse.plot_measure('peak', d=False, percentage=percentage)

        # Plot the 'rawmin' measure in the second subplot
        plt.subplot(3, 1, 2)
        mouse.plot_measure('rawmin', d=False, percentage=percentage)

        # Plot the 'slope' measure in the third subplot
        plt.subplot(3, 1, 3)
        mouse.plot_measure('slope', d=False, percentage=percentage)

        # Set the title of the figure as the mouse ID
        plt.suptitle(mouse.ID)

        # Display the pyplot figure with the measures for the current mouse
        plt.show()


def plt_resize_text(labelsize=18, titlesize=22):
    '''
    Resizes label and title text of the active pyplot figure.

    Args:
        labelsize (int): Font size for axis tick labels. Initialized to 18 as this is generally a good start.
        titlesize (int): Font size for the title. Initialized to 22 as this is generally a good start.
    '''

    # Get the current subplot of the active pyplot figure
    ax = plt.subplot()

    # Set font size for x-axis and y-axis tick labels
    for ticklabel in ax.get_xticklabels():
        ticklabel.set_fontsize(labelsize)
    for ticklabel in ax.get_yticklabels():
        ticklabel.set_fontsize(labelsize)

    # Set font size for x-axis and y-axis labels
    ax.xaxis.get_label().set_fontsize(labelsize)
    ax.yaxis.get_label().set_fontsize(labelsize)

    # Set font size for the title of the figure
    ax.title.set_fontsize(titlesize)



