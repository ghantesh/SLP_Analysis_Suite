# -------------------------------------------------------------------------------------------------------------------------------

# PROJECT - SINGLE LANGMUIR PROBE FITTING SUITE
# MODULE  - PREPROCESSING
# LASTMOD - 05/10/18

# DESCRIPTION: This modules takes the raw signals and extracts important properties for the fitting and plotting modules.
# It also includes some auxiliary functions: lowpass filter, read/write json, etc.

# AKNOWLEDGEMENTS: This project is an ongoing effort to create a practical-generic swept single Langmuir probe analysis suite. Contributions by M Peterka (IPP-CAS), A Devitre (CIEMAT) and A Maan (UTK) are aknowledged.

# -------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt, mlab as mlab
from scipy import signal as sgn
from scipy.ndimage import filters as flt

# -------------------------------------------------------------------------------------------------------------------------------

def lowpass(t, s, cutoff, vb=False):
    '''
    lowpass smoothes the input signal by removing the frequencies greater than cutoff
    implements a gaussian filter
    -------------------------------------
    PARAMS
    t        - (float, array) time axis of signal, in ms (or s)
    s        - (float, array) signal
    cutoff_f - (float) cutoff frequency, in kHz (or Hz) consistent w/ the time axis
    vb       - (bool) optional plots
    -------------------------------------
    RETURNS
    ss       - (float, array) smoothed signal
    '''
    f_sample = (1./(t[1]-t[0]))
    sigma = f_sample/(2.*np.pi*cutoff)
    
    return flt.gaussian_filter1d(s, sigma)


def lowpass_butterworth(t, s, cutoff, vb='False'):
    '''
    lowpass_butterworth smoothes the input signal by removing the frequencies greater than cutoff
    implements a zero-phase Butterworth filter of order 8
    -------------------------------------
    PARAMS
    t        - (float, array) time axis of signal, in ms (or s)
    s        - (float, array) signal
    cutoff_f - (float) cutoff frequency, in kHz (or Hz) consistent w/ the time axis
    vb       - (bool) optional plots
    -------------------------------------
    RETURNS
    ss       - (float, array) smoothed signal
    '''
    f_sample = (1./(t[1]-t[0]))                # sampling frequency
    nyq = 0.5 * f_sample                       # Nyquist rate = 1/2 f_sample
    f_cutoff_norm = cutoff / nyq               # cutoff frequency normalized
    b, a = sgn.butter(8, f_cutoff_norm)        # Butterworth coefficients
    ss = sgn.filtfilt(b, a, s)                 # zero-phase filtering
    
    if vb:                                     # optional plots
        fig1 = plt.figure(figsize=(9,4))
        plt.rc('ytick', labelsize='x-large')
        plt.rc('xtick', labelsize='x-large')
        plt.suptitle('Lowpass verbose')
        ax = plt.gca()
        ax.grid()
        ax.set_xlabel('Time [-]', fontsize='x-large')
        ax.set_ylabel('Amplitude [-]', fontsize='x-large')
        ax.plot(t, s, label='$signal_{raw}$', color = 'black')
        ax.plot(t, ss, label='$signal_{low}, f_{c} = $'+'{}'.format(cutoff), color='green')
        ax.legend(loc = 'upper right', fontsize=24, fancybox=True, framealpha=0.5)
    return ss


def differentiate(t, y):
    '''
    differentiate: returns the first drivative of y along axis t
    the derivative is approximated as (y[i+1]-y[i])/(t[i+1] - t[i])
    -------------------------------------
    PARAMS
    t - (float, array) axis along which the differentiation is carried out
    y - (float, array) array to differentiate
    -------------------------------------
    RETURNS
    y1 (float, array) - derivative of y
    '''
    y1 = np.diff(y)/np.diff(t)
        
    return np.concatenate([y1, [y1[-1]]]) # one less difference than elements in the original array


def add_to_library(shot_number, probe, data):
    '''
    add_to_library creates a new (json file called shotlist.json and) entry with data on each probe/shot
    A convenience to avoid repeating certain calculations (e.g. period and offset)
    -------------------------------------
    PARAMS
    shot_number - (int) shot number
    probe       - (string) probe current-signal name
    data        - (dict) dictionary with data to be stored.
    '''
    try:
        with open('shotlist.json') as f:
            shotlist_dict = json.load(f)
    except IOError:
        shotlist_dict = {}
    except ValueError:
        raise IOError("The JSON file shotlist.json is corrupted.")

    try:
        shotlist_dict[probe][str(shot_number)] = data
    except KeyError:
        shotlist_dict[probe] = {}
        shotlist_dict[probe][str(shot_number)] = data

    try:
        if not(shot_number in shotlist_dict['_shotlist']):
            shotlist_dict['_shotlist'] = shotlist_dict.get('_shotlist', []) + [shot_number]
    except KeyError:
        shotlist_dict['_shotlist'] = shotlist_dict.get('_shotlist', []) + [shot_number]

    with open('shotlist.json', 'w') as f:
        json.dump(shotlist_dict, f, indent=2, sort_keys=True)
      
    
def load_from_library(shot_number, probeid):
    '''
    load_from_library is a get function for data on probe/shot.
    A convenience to avoid repeating certain calculations (e.g. period and offset)
    -------------------------------------
    PARAMS
    shot_number - (int) shot number
    probe - (string) probe current-signal name
    -------------------------------------
    RETURNS
    (dict) data which has been stored about this signal.
    '''
    try:
        with open("shotlist.json") as f:
            shotlist_dict = json.load(f)
            return shotlist_dict[probeid][str(shot_number)]
    except IOError:
        raise IOError("The JSON file with shot data does not exist.")
    except ValueError:
        raise IOError("The JSON file with shot data is corrupted.")


def analyse_signal(t, I, V, shot_params, vb=False):
    '''
    analyse_signal (i) removes the time offset in the voltage signal, starting the first IV at zero volts;
    (ii) collects useful information on the I and V, including the parameters of th parasitic current, 
    and stores them in a json file
    -------------------------------------
    INPUTS
    t           - (float, array) time axis of signal, in ms (or s)
    I           - (float, array) voltage signal raw, there should be a few sweeps before plasma starts!
    V           - (float, array) current signal raw, there should be a few sweeps before plasma starts!
    shot_params - (dict) { 
                           plasma_start - (float) time at which the plasma starts
                           plasma_end   - (float) time at which the plasma ends
                           shot_number  - (int) shot label
                           probeid      - (string) probe label
                         }
    vb          - (bool) optional plots
    -------------------------------------
    RETURNS
    iv_data.    - (DataFrame) pandas DataFrame object with t, I, V, where the capacitive current has been subtracted.
    '''
    
    if vb: print('Cleanup started...')
    if vb: print('...finding sweep frequency and period')
    Pxx, f = mlab.psd(V, NFFT=8388608, Fs=1./(t[1]-t[0]))
    sweep_f = f[np.argmax(Pxx)]
    sweep_p = 1./sweep_f
    if vb:
        fig1 = plt.figure(figsize=(9.6,5))
        plt.rc('ytick', labelsize='large')
        plt.rc('xtick', labelsize='large')
        plt.suptitle('Sweep frequency verbose')
        ax = plt.gca()
        ax.grid()
        ax.set_xlim(xmin = f[0], xmax = 100.*sweep_f)
        ax.set_xlabel('Frequency [-]', fontsize='large')
        ax.set_ylabel('Power spectral density [-]', fontsize='large')
        ax.semilogy(f, Pxx, label='psd', color = 'black')
        ax.semilogy(sweep_f, np.nanmax(Pxx), markersize=10, Marker = 'o', color='white', markeredgecolor='k', label=r'$f_{sweep} \approx $'+'{0:.4f}'.format(sweep_f))
        ax.legend(loc = 'upper right', fontsize=18, fancybox=True, framealpha=0.5)
    
    crop = (shot_params['plasma_start'] - 10.*sweep_p < t) & (t < shot_params['plasma_end'] + 10.*sweep_p) 
    t, I, V = t[crop], I[crop], V[crop]
    
    if vb: print('...finding voltage extrema and trimming to have full IVs')
    V_smooth = lowpass(t, V, 250.*sweep_f, vb=False)
    Vp1 = V_smooth[(t[0] + sweep_p > t) & (t > t[0])]
    tp1 = t[(t[0] + sweep_p > t) & (t > t[0])]
    idx_max, idx_min = np.argmax(Vp1), np.argmin(Vp1)
    vmax, vmin, tmax, tmin = Vp1[idx_max], Vp1[idx_min], tp1[idx_max], tp1[idx_min]
    t0 = np.nanmin([tmin, tmax])
    I, V, V_smooth = I[t > t0], V[t > t0], V_smooth[t > t0]
    t = t[t > t0]
    
    if vb:
        fig2 = plt.figure(figsize=(9.6,5))
        plt.rc('ytick', labelsize='large')
        plt.rc('xtick', labelsize='large')
        plt.suptitle('Extrema verbose')
        ax = plt.gca()
        ax.grid()
        ax.set_xlabel('Time [-]', fontsize='large')
        ax.set_ylabel('Voltage [-]', fontsize='large')
        ax.set_xlim(xmin = t[0]-sweep_p, xmax = t[0]+3*sweep_p)
        ax.plot(t, V, label='V', color = 'green')
        ax.plot(tmax, vmax, markersize=10, Marker = 'o', color='red', markeredgecolor='k', label='$V_{max}$', linestyle='None')
        ax.plot(tmin, vmin, markersize=10, Marker = 'o', color='blue', markeredgecolor='k', label='$V_{min}$', linestyle='None')
        ax.legend(loc = 'upper right', fontsize=18, fancybox=True, framealpha=0.5)
    
    if vb: print('...finding parasitic current parameters ~ C * d(V)/dt << phi + D')
    signal_params = {
        'sweep_f': sweep_f,
        'sweep_p': sweep_p,
        'offset': t0,
        'vmax': np.nanmax([vmax, vmin]),
        'vmin': np.nanmin([vmax, vmin]),
        'plasma_start': shot_params['plasma_start'], 
        'plasma_end': shot_params['plasma_end']
    }
    C, D, phi = parasitic_current(t, I, V, signal_params, vb=vb)
    signal_params.update({'C': C, 'D': D, 'phi': phi}) 
    
    if vb: print('...recording signal parameters in json library')
    add_to_library(shot_params['shot_number'], probe=shot_params['probeid'], data=signal_params)
    if vb: print('...process COMPLETE!')

 
def parasitic_current(t, I, V, signal_params, vb=False):
    '''
    parasitic_current computes a parameterization of the capacitive current ~ C * d(V)/dt << phi + D
    -------------------------------------
    INPUTS
    t             - (float, array) time axis of signal, in ms (or s)
    I             - (float, array) voltage signal raw, there should be a few sweeps before plasma starts!
    V             - (float, array) current signal raw, there should be a few sweeps before plasma starts!
    signal_params - (dict) { 
                             sweep_f      - (float) sweep frequency, units consistent with time axis
                             sweep_p      - (float) sweep period, units consistent with time axis
                             plasma_start - (float) time at which the plasma starts
                             plasma_end   - (float) time at which the plasma ends
                           }
    vb            - (bool) optional plots
    -------------------------------------
    RETURNS
    C             - (float) coupling constant
    D             - (float) current offset
    phi           - (int) phase shift
    '''
    end_of_last_period_before_plasma = signal_params['plasma_start']-(signal_params['plasma_start']%signal_params['sweep_p']) # Keep only sweeps before discharge
    V_noplasma = V[t < end_of_last_period_before_plasma] 
    I_noplasma = I[t < end_of_last_period_before_plasma]
    t_noplasma = t[t < end_of_last_period_before_plasma]
                             
    V1_noplasma = differentiate(t_noplasma, V_noplasma)                                     # differentiate wrt time
    V1_noplasma = lowpass(t_noplasma, V1_noplasma, 100.*signal_params['sweep_f'], vb=False) # smooth dV/dt
    I_noplasma = lowpass(t_noplasma, I_noplasma, 100.*signal_params['sweep_f'], vb=False)   # smooth current

    V1_noplasma = V1_noplasma[(t_noplasma[-1] - signal_params['sweep_p'] > t_noplasma) & (t_noplasma > t_noplasma[0]+ signal_params['sweep_p'])]
    I_noplasma = I_noplasma[(t_noplasma[-1] - signal_params['sweep_p'] > t_noplasma) & (t_noplasma > t_noplasma[0]+ signal_params['sweep_p'])]
    t_noplasma = t_noplasma[(t_noplasma[-1] - signal_params['sweep_p'] > t_noplasma) & (t_noplasma > t_noplasma[0]+ signal_params['sweep_p'])]
    
    D = np.nanmean(I_noplasma)                          # current offset
    C = np.nanmax(I_noplasma-D)/np.nanmax(V1_noplasma)  # ~ max-amplitude-V'/max-amplitud-I
    V1_noplasma *= C
    V1_noplasma += D
    
    correlation = sgn.correlate((I_noplasma-np.mean(I_noplasma))/np.std(I_noplasma), (V1_noplasma-np.mean(V1_noplasma))/np.std(V1_noplasma))
    phi = np.arange(1-len(I_noplasma), len(I_noplasma))[np.argmax(correlation)]
    V1_noplasma = np.roll(V1_noplasma, phi)

    if vb:
        fig1 = plt.figure(figsize=(9.6,5))
        plt.rc('ytick', labelsize='large')
        plt.rc('xtick', labelsize='large')
        plt.suptitle('Parasitic current verbose')
        ax = plt.gca()
        ax.grid()
        ax.set_xlabel('Time [-]', fontsize='large')
        ax.set_ylabel('Current [-]', fontsize='large')
        ax.set_xlim(xmin = t_noplasma[0], xmax = t_noplasma[-1])
        ax.plot(t_noplasma, I_noplasma, linewidth=2, Marker='*', linestyle='None', color = 'gray', alpha=0.5, label=r'$I_{no plasma}$')
        ax.plot(t_noplasma, V1_noplasma, Marker='o', linestyle='None', color = 'red', label=r'$\tilde{I} \approx $'+'{0:.2E}'.format(C)+r'$\frac{\partial V}{\partial t}$'+'{0:.2E}'.format(D))
        ax.legend(loc = 'upper right', fontsize=18, fancybox=True, framealpha=0.5)
    
    return C, D, int(phi) # np.int64 are not JSON serializable.


def prepare_signal(t, I, V, shot_number, probeid, vb='False'):
    '''
    prepare_signal removes the capacitive current and loads the signal info data
    -------------------------------------
    INPUTS
    shot_number - (int) shot number
    probe       - (string) probe current-signal name
    vb          - (bool) optional plots
    -------------------------------------
    RETURNS
    signal_params - (dict) { 
                             sweep_f      - (float) sweep frequency, units consistent with time axis
                             sweep_p      - (float) sweep period, units consistent with time axis
                             plasma_start - (float) time at which the plasma starts
                             plasma_end   - (float) time at which the plasma ends
                             offset       - (float) time of first sweep minimum
                             vmax         - (float) maximum sweep voltage
                             vmin         - (float) minimum sweep voltage
                             C            - (float) parasitic current: coupling constant
                             D            - (float) parasitic current: offset current
                             phi          - (float) parasitic current: phase (in pts)
                           }
    iv_data.      - (dict) {
                             t            - (float, array) time axis of signal, in ms (or s)
                             I            - (float, array) current signal (parasitic current removed)
                             V            - (float, array) voltage signal raw
                           }
    '''
    # load signals [ANURAG ->  correct call]
    
    # load the signal info data
    signal_params = load_from_library(shot_number=shot_number, probeid=probeid)
    
    # start first IV at minimum voltage
    crop = (signal_params['plasma_end']+10.*signal_params['sweep_p'] > t) & (t > signal_params['offset'])
    I = I[crop]
    V = V[crop]
    t = t[crop]
    
    # remove parasitic current
    V1 = lowpass(t, differentiate(t, V), 100.*signal_params['sweep_f'], vb=False)
    I_clean = I - (signal_params['C'] * np.roll(V1, signal_params['phi']) + signal_params['D'])
    
    # create an IV-index
    n = 1
    absolute_time = t-t[0]
    sweep_p = signal_params['sweep_p']
    sweep_numbers = np.zeros_like(t, dtype='int')

    for i, time in enumerate(absolute_time):
        if time > n*sweep_p:
            n+=1
        sweep_numbers[i] = n
    
    if vb:
        fig1 = plt.figure(figsize=(9.6,5))
        plt.rc('ytick', labelsize='large')
        plt.rc('xtick', labelsize='large')
        plt.suptitle('Prep verbose')
        ax = plt.gca()
        ax.grid()
        ax.set_xlabel('Time [-]', fontsize='large')
        ax.set_ylabel('Current [-]', fontsize='large')
        ax.set_xlim(xmin = t[0], xmax = t[-1])
        ax.plot(t, I, linewidth=2, Marker='*', linestyle='None', color = 'k', alpha=1, label=r'$I_{raw}$')
        ax.plot(t, I_clean, Marker='o', linestyle='None', color = 'red', alpha=0.2, label=r'$I_{clean}$')
        ax.legend(loc = 'upper right', fontsize=18, fancybox=True, framealpha=0.5)
        
    return signal_params, pd.DataFrame.from_dict({'t': t, 'sweep_no': sweep_numbers, 'I': I_clean, 'V': V})