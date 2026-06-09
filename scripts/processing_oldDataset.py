import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# # importing libraries ____________________________________________________________________________________________________________


import mat73
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from tqdm import tqdm
import py_utils.signal_processing as sgnpr
import py_utils.data_managment as dtmn
from py_utils.eeg_managment import proc_pos2win
import riemann_utils.covariances as rutils 
import riemann_utils.matrix_functions as mrtrf

def processDataset_20192020(pathData, doLogBandPower, bandranges, filter_order, windowsLength, windowsShift, saveData=False, saveName='', applyLaplacian=True, fs=512):
    
    # # loading dataset ______________________________________________________________________________________________________________
    if isinstance(pathData, str):
        file_path = f'{pathData}dataset_user_20192020.mat'
        dataset = mat73.loadmat(file_path)
    else:
        dataset = pathData

    print('- Dataset loaded')

    dataset = dataset['data']
    eeg = dataset['eeg']
    ev = dataset['events']       
    labels = dataset['labels']

    # # removing day mean for the logbandpower ________________________________________________________________________________________
    if doLogBandPower:
        for day in np.unique(labels['Dk']):
            eeg_day = eeg[labels['Dk'] == day]
            eeg[labels['Dk'] == day] = eeg_day - np.mean(eeg_day, axis=0)

    # # processing events ____________________________________________________________________________________________________________
    events = pd.DataFrame(ev)
    unique_events = np.unique(events.TYP)
    off_events = unique_events[unique_events>0x8000]

    if 781+0x8000 in off_events:
        off = 781+0x8000
        on = off-0x8000
        idx_off_ev = events[events.TYP==off].index.to_numpy()
        idx_on_ev = events[(events.TYP==on) & (events.DUR==0)].index.to_numpy()
        pos_on = events.POS[idx_on_ev].values
        for i in idx_off_ev:
            pos_off = events.POS[i]
            idx = np.where(pos_on<pos_off)[0][-1]
            ev_idx = idx_on_ev[idx]
            events.DUR[ev_idx] = pos_off-pos_on[idx]

    if 1024+0x8000 in off_events:
        eog_second = 2
        dur = eog_second*fs
        events.DUR[events.TYP==1024] = dur

    events.drop(events[events.TYP>0x8000].index, inplace=True)
    events.DUR[np.isnan(events.DUR)] = 0 

    events_code = {'hands':         773,
               'feet':          771,
               'PadLeft':       201,
               'PadLight':      202,
               'PadRight':      203,
               'PadNone':       204,
               'CommandLeft':   101,
               'CommandLight':  102,
               'CommandRight':  103,
               'cont_fdback':   781,
               'fixation':      786,
               'race_start':    800,
               'hit':           897,
               'miss':          898,
               'eog':           1024,
               '2020_Right':    1670,
               '2020_Left':     1672}

    dict_lambda = {events_code['hands']:        events_code['hands'],
                events_code['feet']:         events_code['feet'],
                events_code['PadLeft']:      events_code['hands'],
                events_code['PadLight']:     np.nan,
                events_code['PadRight']:     events_code['feet'],
                events_code['PadNone']:      np.nan,
                events_code['CommandLeft']:  events_code['hands'],
                events_code['CommandLight']: np.nan,
                events_code['CommandRight']: events_code['feet'],
                events_code['cont_fdback']:  events_code['cont_fdback'],
                events_code['fixation']:     events_code['fixation'],
                events_code['race_start']:   np.nan,
                events_code['hit']:          events_code['hit'],
                events_code['miss']:         events_code['miss'],
                events_code['eog']:          events_code['eog'],
                events_code['2020_Right']:   events_code['feet'],
                events_code['2020_Left']:    events_code['hands']}

    events['TYP'] = events['TYP'].apply(lambda x: dict_lambda[x])
    events.drop(events[np.isnan(events['TYP'])].index, inplace=True)
    events.reset_index(drop=True, inplace=True)
    events.columns = events.columns.str.lower()


    # # processing channels ____________________________________________________________________________________________________________
    ch_labels = {0:np.array(['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',  'C3',  'C1', 'CZ',  'C2',  'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4' ]),
                      1:np.array(['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',  'C3',  'C1','CZ',  'C2',  'C4', 'EOG', 'CP1', 'CPZ', 'CP2', 'EOG' ])}

    channels_to_remove = ['FZ']

    _,idx,_ = np.intersect1d(ch_labels[0],ch_labels[1],return_indices=True)
    idx.sort()
    channels = ch_labels[0][idx]

    idx_ch = [list(range(0,len(ch_labels[0]))), idx]
    laplacian_mask = []

    path = '/home/palatella/workspace/cap_utils/laplacian16.mat'
    lap = loadmat(path)
    laplacian_mask.append(lap['lapmask'])

    path = '/home/palatella/workspace/cap_utils/laplacian16_IntrscEOG.mat'
    lap = loadmat(path)
    laplacian_mask.append(lap['lapmask'])

    idx_final_eeg = idx
    _,idx,_ = np.intersect1d(ch_labels[0],channels_to_remove,return_indices=True)
    idx_final_eeg = np.delete(idx_final_eeg,idx)
    channels = np.delete(channels,idx)


    # # processing eeg  _______________________________________________________________________________________________________________
    for eog_idx in ch_labels.keys():
        idx_eog = labels['EOGk']==eog_idx
        if applyLaplacian: 
            print('Applying laplacian')
            eeg[np.ix_(idx_eog, idx_ch[eog_idx])] = eeg[np.ix_(idx_eog, idx_ch[eog_idx])] @ laplacian_mask[eog_idx]

    eeg_lap_old = eeg[:,idx_final_eeg]

    eeg_bands = sgnpr.get_bandranges(eeg_lap_old, bandranges, fs, filter_order, filtType='bandpass')
    # eeg_bands = array_with_new_dim = np.expand_dims(eeg, axis=0)
    n_bandranges, nsamples, nchannels = eeg_bands.shape

    if doLogBandPower:
        for bId in range(n_bandranges):
            eeg_bands[bId] = sgnpr.logbandpower(eeg_bands[bId], fs = fs, slidingWindowLength=1)


    # # getting covariance and label vectors  _____________________________________________________________________________________________
    nwindows = 0
    for k in np.unique(labels['Rk']):
        nsamples = len(np.where(labels['Rk']==k)[0])
        nwindows += int((nsamples-windowsLength*fs)/(windowsShift*fs))+1

    covs = np.full((n_bandranges, nwindows, nchannels, nchannels), np.nan)
    cov_events = pd.DataFrame(columns=events.columns)
    runVector = np.full((nwindows), np.nan)
    eogVector = np.full((nwindows), np.nan)
    dayVector = np.full((nwindows), np.nan) 
    modalityVector = np.full((nwindows), np.nan)
    protocolVector = np.full((nwindows), np.nan)

    idx_shift = 0
    for k in tqdm (np.unique(labels['Rk']), bar_format='{l_bar}{bar:40}{r_bar}'):
        idx_run = np.where(labels['Rk']==k)[0]
        t_events = events[events.run==k].reset_index(drop=True)
        t_events.loc[:,'pos'] = t_events.pos.values - 1 -(np.min(idx_run) if k>1 else 0)
        [t_covs,t_cov_events] = sgnpr.get_trNorm_covariance_matrix(eeg_bands[:,idx_run], t_events, windowsLength, windowsShift, fs, dispProgress=False)
        covs[:,idx_shift:idx_shift+t_covs.shape[1],:,:] = t_covs

        t_cov_events.pos = t_cov_events.pos + idx_shift
        cov_events = pd.concat([cov_events, t_cov_events], axis=0, ignore_index=True)

        runVector[idx_shift:idx_shift+t_covs.shape[1]] = k
        if k==137:  # 137th run without events (from data)
            eogVector[idx_shift:idx_shift+t_covs.shape[1]] = eogVector[runVector==k-1][0]   #fai il giorno prima
            dayVector[idx_shift:idx_shift+t_covs.shape[1]] = dayVector[runVector==k-1][0]  
        else:
            eogVector[idx_shift:idx_shift+t_covs.shape[1]] = t_cov_events.loc[0,'eog']
            dayVector[idx_shift:idx_shift+t_covs.shape[1]] = t_cov_events.loc[0,'day']
        modalityVector[idx_shift:idx_shift+t_covs.shape[1]] = labels['Mk'][idx_run[0]]
        protocolVector[idx_shift:idx_shift+t_covs.shape[1]] = labels['Pk'][idx_run[0]]
        
        idx_shift += t_covs.shape[1]

    cov_events = cov_events.astype(int)
    utilsVector = {'run': runVector, 'eog': eogVector, 'day': dayVector, 'modality': modalityVector, 'protocol': protocolVector, 'daysLabel':labels['Dl']}

    if saveData:
        if saveName == '':      saveName = 'covs_user.mat' if not doLogBandPower else 'covs_logBandPower_user.mat'
        print('Saving data into ' + f'{pathData}{saveName} ...')
        savemat(f'{pathData}{saveName}',{'covs':covs,'cov_events':cov_events.to_numpy(), 'column_names': cov_events.columns.values, 'utilsVector': utilsVector})
    else:
        print('Data not saved')




def processDataset_calibrations(pathData, days_modelCreation, bandranges, filter_order, windowsLength, windowsShift, doLogBandPower, saveName='', applyLaplacian=True, saveData=False, fs=512):
    pathDataCal = '/home/palatella/workspace/cybathlon_user/calibrations/'

    covs = np.empty(len(days_modelCreation), dtype=object)
    cov_events = np.empty(len(days_modelCreation), dtype=object)

    
    # # processing channels ____________________________________________________________________________________________________________
    ch_labels = {0:np.array(['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',  'C3',  'C1', 'CZ',  'C2',  'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4' ]),
                      1:np.array(['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',  'C3',  'C1','CZ',  'C2',  'C4', 'EOG', 'CP1', 'CPZ', 'CP2', 'EOG' ])}

    channels_to_remove = ['FZ']

    _,idx,_ = np.intersect1d(ch_labels[0],ch_labels[1],return_indices=True)
    idx.sort()

    path = '/home/palatella/workspace/cap_utils/laplacian16.mat'
    laplacian_mask = loadmat(path)

    idx_final_eeg = idx
    _,idx,_ = np.intersect1d(ch_labels[0],channels_to_remove,return_indices=True)
    idx_final_eeg = np.delete(idx_final_eeg,idx)



    for idx,day in enumerate(days_modelCreation):
        date = day[-4:] + day[3:5] + day[:2]
        filenames = dtmn.get_filesNames_from_folder(pathDataCal, date)
        filenames = [pathDataCal + fl for fl in filenames]
        signals, events = dtmn.load_gdf_files(filenames)

        # # removing day mean for the logbandpower
        if doLogBandPower:  signals = signals - np.mean(signals, axis=0)
        
        if applyLaplacian: signals = signals @ laplacian_mask['lapmask']
        

        signals = signals[:,idx_final_eeg]

        eeg_bands = sgnpr.get_bandranges(signals, bandranges, fs, filter_order)
        n_bandranges, nsamples, nchannels = eeg_bands.shape

        if doLogBandPower:
            for bId in range(n_bandranges):
                eeg_bands[bId] = sgnpr.logbandpower(eeg_bands[bId], fs = fs, slidingWindowLength=1)

        [covs[idx],t_cov_events] = sgnpr.get_trNorm_covariance_matrix(eeg_bands, events, windowsLength, windowsShift, fs, dispProgress=False)

        column_names = t_cov_events.columns.values
        cov_events[idx] = t_cov_events.to_numpy() 


    if saveData:
        if saveName == '':  saveName = 'covs_modelCal_user.mat' if not doLogBandPower else 'covs_modelCal_logBandPower_user.mat'
        print('Saving data into ' + f'{pathData}{saveName} ...')
        savemat(f'{pathData}{saveName}',{'covs':covs,'cov_events':cov_events, 'column_names': column_names})
    else:
        print('Data not saved') 




def extract_model_centroids(covs, cov_events, classes, pathData, doLogBandPower, saveData=False, mean_cov_first=None, inv_sqrt_mean_cov_first=None, saveName=''):

    model_centroids = np.empty((covs.shape[0]), dtype=object)
    std_Mdlcentroids = np.empty((covs.shape[0]), dtype=object)
    absDev_Mdlcentroids = np.empty((covs.shape[0]), dtype=object)
    total_model_centroids = np.empty((covs.shape[0]), dtype=object)
    total_std_Mdlcentroids = np.empty((covs.shape[0]), dtype=object)
    total_absDev_Mdlcentroids = np.empty((covs.shape[0]), dtype=object)

    for n_day in range(covs.shape[0]):
        
        t_covs = covs[n_day]
        t_events = cov_events[n_day]
        isCFeedback = np.full((t_covs.shape[1]), False)
        Ck = np.full((t_covs.shape[1]), 0)

        model_centroids[n_day] = np.full((len(classes), t_covs.shape[0], t_covs.shape[-2], t_covs.shape[-1]), np.nan)
        std_Mdlcentroids[n_day] = np.full((len(classes), t_covs.shape[0]), np.nan)
        absDev_Mdlcentroids[n_day] = np.full((len(classes), t_covs.shape[0]), np.nan)

        total_model_centroids[n_day] = np.full((t_covs.shape[0], t_covs.shape[-2], t_covs.shape[-1]), np.nan)
        total_std_Mdlcentroids[n_day] = np.full((t_covs.shape[0]), np.nan)
        total_absDev_Mdlcentroids[n_day] = np.full((t_covs.shape[0]), np.nan)

        for idx, cfRow in t_events[t_events.typ==781].iterrows():
            isCFeedback[cfRow.pos:cfRow.pos+cfRow.dur] = True
            cue = t_events.loc[idx-1,'typ']
            if cue in classes:  Ck[cfRow.pos:cfRow.pos+cfRow.dur] = cue

        if mean_cov_first is not None: t_covs = rutils.center_covariances(t_covs, mean_cov_first, inv_sqrt_mean_cov_first)

        for idx, clss in enumerate(classes):   
            mean_cov, _ = rutils.get_riemann_mean_covariance(t_covs[:,Ck==clss], show_progess=False, print_print=False) 
            model_centroids[n_day][idx], _ = mean_cov
            std_Mdlcentroids[n_day][idx] = mrtrf.matrix_std(t_covs[:,Ck==clss], mean_cov)
            absDev_Mdlcentroids[n_day][idx] = mrtrf.matrix_meanAbsoluteDeviation(t_covs[:,Ck==clss], mean_cov)

        mean_cov, _ = rutils.get_riemann_mean_covariance(t_covs[:,isCFeedback], show_progess=False, print_print=False)  
        total_model_centroids[n_day], _ = mean_cov
        total_std_Mdlcentroids[n_day] = mrtrf.matrix_std(t_covs[:,isCFeedback], mean_cov)
        total_absDev_Mdlcentroids[n_day] = mrtrf.matrix_meanAbsoluteDeviation(t_covs[:,isCFeedback], mean_cov)

    if saveData:
        if saveName == '':  saveName = 'centroids_modelCal_user.mat' if not doLogBandPower else 'centroids_modelCal_logBandPower_user.mat'
        print('Saving data into ' + f'{pathData}{saveName} ...')
        savemat(f'{pathData}{saveName}',{'model_centroids':model_centroids,'total_model_centroids':total_model_centroids, 'std_Mdlcentroids':std_Mdlcentroids, 'absDev_Mdlcentroids':absDev_Mdlcentroids,
                       'total_std_Mdlcentroids':total_std_Mdlcentroids, 'total_absDev_Mdlcentroids':total_absDev_Mdlcentroids}) 
    else:
        print('Data not saved') 
    
    
             


def processDataset_PSD_20192020(pathData, internalWindowsLength, windowsShift, pshift, mlength, fs=512):
    print('- Processing PSD dataset 2019-2020')
    # # loading dataset ______________________________________________________________________________________________________________
    if isinstance(pathData, str):
        file_path = f'{pathData}dataset_user_20192020.mat'
        dataset = mat73.loadmat(file_path)
    else:
        dataset = pathData

    print('- Dataset loaded')

    dataset = dataset['data']
    eeg = dataset['eeg']
    ev = dataset['events']       
    labels = dataset['labels']

    # # processing events ____________________________________________________________________________________________________________
    events = pd.DataFrame(ev)
    unique_events = np.unique(events.TYP)
    off_events = unique_events[unique_events>0x8000]

    if 781+0x8000 in off_events:
        off = 781+0x8000
        on = off-0x8000
        idx_off_ev = events[events.TYP==off].index.to_numpy()
        idx_on_ev = events[(events.TYP==on) & (events.DUR==0)].index.to_numpy()
        pos_on = events.POS[idx_on_ev].values
        for i in idx_off_ev:
            pos_off = events.POS[i]
            idx = np.where(pos_on<pos_off)[0][-1]
            ev_idx = idx_on_ev[idx]
            events.DUR[ev_idx] = pos_off-pos_on[idx]

    if 1024+0x8000 in off_events:
        eog_second = 2
        dur = eog_second*fs
        events.DUR[events.TYP==1024] = dur

    events.drop(events[events.TYP>0x8000].index, inplace=True)
    events.DUR[np.isnan(events.DUR)] = 0 

    events_code = {'hands':         773,
               'feet':          771,
               'PadLeft':       201,
               'PadLight':      202,
               'PadRight':      203,
               'PadNone':       204,
               'CommandLeft':   101,
               'CommandLight':  102,
               'CommandRight':  103,
               'cont_fdback':   781,
               'fixation':      786,
               'race_start':    800,
               'hit':           897,
               'miss':          898,
               'eog':           1024,
               '2020_Right':    1670,
               '2020_Left':     1672}

    dict_lambda = {events_code['hands']:        events_code['hands'],
                events_code['feet']:         events_code['feet'],
                events_code['PadLeft']:      events_code['hands'],
                events_code['PadLight']:     np.nan,
                events_code['PadRight']:     events_code['feet'],
                events_code['PadNone']:      np.nan,
                events_code['CommandLeft']:  events_code['hands'],
                events_code['CommandLight']: np.nan,
                events_code['CommandRight']: events_code['feet'],
                events_code['cont_fdback']:  events_code['cont_fdback'],
                events_code['fixation']:     events_code['fixation'],
                events_code['race_start']:   np.nan,
                events_code['hit']:          events_code['hit'],
                events_code['miss']:         events_code['miss'],
                events_code['eog']:          events_code['eog'],
                events_code['2020_Right']:   events_code['feet'],
                events_code['2020_Left']:    events_code['hands']}

    events['TYP'] = events['TYP'].apply(lambda x: dict_lambda[x])
    events.drop(events[np.isnan(events['TYP'])].index, inplace=True)
    events.reset_index(drop=True, inplace=True)
    events.columns = events.columns.str.lower()


    # # processing channels ____________________________________________________________________________________________________________
    ch_labels = {0:np.array(['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',  'C3',  'C1', 'CZ',  'C2',  'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4' ]),
                      1:np.array(['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',  'C3',  'C1','CZ',  'C2',  'C4', 'EOG', 'CP1', 'CPZ', 'CP2', 'EOG' ])}

    channels_to_remove = ['FZ']

    _,idx,_ = np.intersect1d(ch_labels[0],ch_labels[1],return_indices=True)
    idx.sort()
    channels = ch_labels[0][idx]

    idx_ch = [list(range(0,len(ch_labels[0]))), idx]
    laplacian_mask = []

    path = '/home/palatella/workspace/cap_utils/laplacian16.mat'
    lap = loadmat(path)
    laplacian_mask.append(lap['lapmask'])

    path = '/home/palatella/workspace/cap_utils/laplacian16_IntrscEOG.mat'
    lap = loadmat(path)
    laplacian_mask.append(lap['lapmask'])

    # idx_final_eeg = idx
    # _,idx,_ = np.intersect1d(ch_labels[0],channels_to_remove,return_indices=True)
    # idx_final_eeg = np.delete(idx_final_eeg,idx)
    # channels = np.delete(channels,idx)


    # # processing eeg  _______________________________________________________________________________________________________________
    print('- Applying laplacian')
    for eog_idx in ch_labels.keys():
        idx_eog = labels['EOGk']==eog_idx
        eeg[np.ix_(idx_eog, idx_ch[eog_idx])] = eeg[np.ix_(idx_eog, idx_ch[eog_idx])] @ laplacian_mask[eog_idx]

    # eeg_lap = eeg[:,idx_final_eeg]


    ## load classifiers  _____________________________________________________________________________________________
    channelBandDict = {}
    models_path = '/home/palatella/workspace/f1_classifiers/'

    files = sorted([f for f in os.listdir(models_path) if ('2019' in f or '2020' in f)])
    models_paths = [os.path.join(models_path, f) for f in files]

    for f in models_paths:
        m = dtmn.fix_mat(loadmat(f)['settings'])
        bands = m['bci']['smr']['bands']
        
        for b in range(len(bands)):
            bands[b] = np.array(bands[b]).squeeze()
        
        non_empty_idx = [i for i, b in enumerate(bands) if b.size > 0]

        for idx in non_empty_idx:
            band = np.array(bands[idx]).squeeze()

            if idx not in channelBandDict:   channelBandDict[idx] = set()  

            if band.size == 2:      channelBandDict[idx].update(band)
            else:                   channelBandDict[idx].update(int(val) for val in np.atleast_1d(band).flatten())

    # convert sets to sorted lists
    channelBandDict = {k: sorted(v) for k, v in channelBandDict.items()}
    channelIdx = sorted(channelBandDict.keys())
    # channels = ch_labels[0][channelIdx]

    eeg_selected = eeg[:, channelIdx]
    nchannels = len(channelIdx)
    frequencies = np.unique([val for sublist in channelBandDict.values() for val in sublist]).astype(int)



    # # getting psd and label vectors  _____________________________________________________________________________________________
    nwindows = 0
    for k in np.unique(labels['Rk']):
        nsamples = len(np.where(labels['Rk']==k)[0])
        nwindows += int((nsamples-mlength*fs)/(windowsShift*fs))+1


    psd = np.full((nwindows, len(frequencies), nchannels), np.nan)
    psd_events = pd.DataFrame(columns=events.columns)
    runVector = np.full((nwindows), np.nan)
    eogVector = np.full((nwindows), np.nan)
    dayVector = np.full((nwindows), np.nan) 
    modalityVector = np.full((nwindows), np.nan)
    protocolVector = np.full((nwindows), np.nan)

    idx_shift = 0
    for k in tqdm (np.unique(labels['Rk']), bar_format='{l_bar}{bar:40}{r_bar}'):
        idx_run = np.where(labels['Rk']==k)[0]
        t_events = events[events.run==k].reset_index(drop=True)
        t_events.loc[:,'pos'] = t_events.pos.values - 1 - (np.min(idx_run) if k>1 else 0)
        t_psd, freq = sgnpr.proc_spectrogram(eeg_selected[idx_run], internalWindowsLength, windowsShift, pshift, fs, mlength=mlength)

        t_events['pos'] = proc_pos2win(t_events['pos'], windowsShift*fs, 'backward', mlength*fs)
        t_events['dur'] = [ int(x) for x in t_events['dur']/(windowsShift*fs)+1 ]

        psd[idx_shift:idx_shift+t_psd.shape[0],:,:] = t_psd[:, np.isin(freq, frequencies), :]

        t_events.pos = t_events.pos + idx_shift
        psd_events = pd.concat([psd_events, t_events], axis=0, ignore_index=True)

        runVector[idx_shift:idx_shift+t_psd.shape[0]] = k
        if k==137:  # 137th run without events (from data)
            eogVector[idx_shift:idx_shift+t_psd.shape[0]] = eogVector[runVector==k-1][0]   #fai il giorno prima
            dayVector[idx_shift:idx_shift+t_psd.shape[0]] = dayVector[runVector==k-1][0]  
        else:
            eogVector[idx_shift:idx_shift+t_psd.shape[0]] = t_events.loc[0,'eog']
            dayVector[idx_shift:idx_shift+t_psd.shape[0]] = t_events.loc[0,'day']
        modalityVector[idx_shift:idx_shift+t_psd.shape[0]] = labels['Mk'][idx_run[0]]
        protocolVector[idx_shift:idx_shift+t_psd.shape[0]] = labels['Pk'][idx_run[0]]
        
        idx_shift += t_psd.shape[0]

    psd = np.log(psd)

    return psd, freq[np.isin(freq, frequencies)], psd_events, runVector, eogVector, dayVector, modalityVector, protocolVector, channelBandDict

    
    
def get_FisherScore_perDay(eeg, labels, events, internalWindowsLength, windowsShift, pshift, mlength, fs=512):
    print('- Computing Fisher Score per day')
    eeg_selected = eeg #[:, channelIdx]
    nchannels = eeg_selected.shape[1]
    # frequencies = np.unique([val for sublist in channelBandDict.values() for val in sublist]).astype(int)

    # # getting psd and label vectors  _____________________________________________________________________________________________
    nwindows = 0
    for k in np.unique(labels['Rk']):
        nsamples = len(np.where(labels['Rk']==k)[0])
        nwindows += int((nsamples-mlength*fs)/(windowsShift*fs))+1

    # idx_shift = 0
    fisherScore = np.empty((len(np.unique(labels['Dk']))), dtype=object)
    cva = np.empty((len(np.unique(labels['Dk']))), dtype=object)
    idx_day = 0
    for k in tqdm (np.unique(labels['Dk']), bar_format='{l_bar}{bar:40}{r_bar}'):
        if k<14:
            continue
        idx_run = np.where(labels['Dk']==k)[0]
        selected_runs = np.unique(labels['Rk'][idx_run])
        
        t_events = events[events.run.isin(selected_runs)].reset_index(drop=True)
        t_events.loc[:,'pos'] = t_events.pos.values - 1 - (np.min(idx_run) if k>1 else 0)
        psd, freq = sgnpr.proc_spectrogram(eeg_selected[idx_run], internalWindowsLength, windowsShift, pshift, fs, mlength=mlength)

        psd = np.log(psd)

        t_events['pos'] = proc_pos2win(t_events['pos'], windowsShift*fs, 'backward', mlength*fs)
        t_events['dur'] = [ int(x) for x in t_events['dur']/(windowsShift*fs)+1 ]

        idx_cf = t_events[t_events.typ==781].index.to_numpy()
        
        classes = np.array([c for c in [771,773, 770, 769] if c in np.unique(t_events.loc[idx_cf-1,'typ'].values)])

        runs_labels = np.empty((1), dtype=object)
        lbl = []
        isCFeedbackVector = np.full((psd.shape[0]), False)

        for idx in idx_cf:
            isCFeedbackVector[t_events.loc[idx,'pos']:t_events.loc[idx,'pos']+t_events.loc[idx,'dur']] = True
            lbl.append(t_events.loc[idx-1,'typ'].repeat(t_events.loc[idx,'dur']))

        runs_labels[0] = np.concatenate(lbl)

        selFreq = np.array(range(6,28,2))
        fisherScore[idx_day], cva[idx_day] = sgnpr.compute_fisher_score(psd, freq, classes, np.ones((psd.shape[0])), runs_labels, isCFeedbackVector=isCFeedbackVector, SelFreqs=selFreq)
        idx_day += 1

    return fisherScore, cva, freq



def processDataset_PSD_calibrations(days_modelCreation, internalWindowsLength, windowsShift, pshift, mlength, fs=512):
    pathDataCal = '/home/palatella/workspace/cybathlon_user/calibrations/'

    psd = np.empty(len(days_modelCreation), dtype=object)
    psd_events = np.empty(len(days_modelCreation), dtype=object)

    
    # # processing channels ____________________________________________________________________________________________________________
    ch_labels = {0:np.array(['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',  'C3',  'C1', 'CZ',  'C2',  'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4' ]),
                      1:np.array(['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',  'C3',  'C1','CZ',  'C2',  'C4', 'EOG', 'CP1', 'CPZ', 'CP2', 'EOG' ])}

    channels_to_remove = ['FZ']

    _,idx,_ = np.intersect1d(ch_labels[0],ch_labels[1],return_indices=True)
    idx.sort()

    path = '/home/palatella/workspace/cap_utils/laplacian16.mat'
    laplacian_mask = loadmat(path)

    idx_final_eeg = idx
    _,idx,_ = np.intersect1d(ch_labels[0],channels_to_remove,return_indices=True)
    idx_final_eeg = np.delete(idx_final_eeg,idx)

    for idxDay,day in enumerate(days_modelCreation):
        date = day[-4:] + day[3:5] + day[:2]
        filenames = dtmn.get_filesNames_from_folder(pathDataCal, date)
        filenames = [pathDataCal + fl for fl in filenames]
        eeg, events, _ = dtmn.load_gdf_files(filenames)

        eeg = eeg @ laplacian_mask['lapmask']


        ## load classifiers  _____________________________________________________________________________________________
        channelBandDict = {}
        models_path = '/home/palatella/workspace/f1_classifiers/'

        files = sorted([f for f in os.listdir(models_path) if ('2019' in f or '2020' in f)])
        models_paths = [os.path.join(models_path, f) for f in files]

        for f in models_paths:
            m = dtmn.fix_mat(loadmat(f)['settings'])
            bands = m['bci']['smr']['bands']
            
            for b in range(len(bands)):
                bands[b] = np.array(bands[b]).squeeze()
            
            non_empty_idx = [i for i, b in enumerate(bands) if b.size > 0]

            for idx in non_empty_idx:
                band = np.array(bands[idx]).squeeze()

                if idx not in channelBandDict:   channelBandDict[idx] = set()  

                if band.size == 2:      channelBandDict[idx].update(band)
                else:                   channelBandDict[idx].update(int(val) for val in np.atleast_1d(band).flatten())


        # convert sets to sorted lists
        channelBandDict = {k: sorted(v) for k, v in channelBandDict.items()}
        channelIdx = sorted(channelBandDict.keys())
        # channels = ch_labels[0][channelIdx]
        eeg_selected = eeg[:, channelIdx]
     
        frequencies = np.unique([val for sublist in channelBandDict.values() for val in sublist]).astype(int)
        
        t_psd, freq = sgnpr.proc_spectrogram(eeg_selected, internalWindowsLength, windowsShift, pshift, fs, mlength=mlength)

        events['pos'] = proc_pos2win(events['pos'], windowsShift*fs, 'backward', mlength*fs)
        events['dur'] = [ int(x) for x in events['dur']/(windowsShift*fs)+1 ]

        psd[idxDay] = np.log(t_psd[:, np.isin(freq, frequencies), :])
        psd_events[idxDay] = events

    return psd, freq[np.isin(freq, frequencies)], psd_events, channelBandDict
