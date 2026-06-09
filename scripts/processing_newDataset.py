import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# # importing libraries ____________________________________________________________________________________________________________

from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
from tqdm import tqdm
import py_utils.signal_processing as sgnpr
import py_utils.data_managment as dtmn
from py_utils.eeg_managment import proc_pos2win



def processDataset_20232024(pathData, filename, doLogBandPower, bandranges, filter_order, windowsLength, windowsShift, applyLaplacian=True, saveData=False, saveName='', fs=512):
   
    # # loading dataset ______________________________________________________________________________________________________________
    data = loadmat(f'{pathData}{filename}')
    eeg = data['eeg']
    events = pd.DataFrame(dtmn.fix_mat(data['events']))   
    labels = dtmn.fix_mat(data['procLabels'])
    for key in labels.keys():
        labels[key] = labels[key].astype(int)
    events.columns = events.columns.str.lower()
    events.pos = events.pos - 1

    if 'days' not in data.keys():
        days = np.array(['03/10/2023','11/10/2024','14/10/2024','15/10/2024','18/10/2024','21/10/2024','22/10/2024'])
    else:
        days = np.array([date[0] for date in np.squeeze(data['days'])])


    # # removing day mean for the logbandpower ________________________________________________________________________________________
    if doLogBandPower:
        for day in np.unique(labels['Dk']):
            eeg_day = eeg[labels['Dk'] == day]
            eeg[labels['Dk'] == day] = eeg_day - np.mean(eeg_day, axis=0)


    # # processing channels ____________________________________________________________________________________________________________
    ch_labels = {0:np.array(['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',  'C3',  'C1', 'CZ',  'C2',  'C4', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4' ]),
                      1:np.array(['FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',  'C3',  'C1','CZ',  'C2',  'C4', 'EOG', 'CP1', 'CPZ', 'CP2', 'EOG' ])}

    channels_to_remove = ['FZ']

    _,idx,_ = np.intersect1d(ch_labels[0],ch_labels[1],return_indices=True)
    idx.sort()

    path = '/home/palatella/workspace/cap_utils/laplacian16.mat'
    lap = loadmat(path)

    idx_final_eeg = idx
    _,idx,_ = np.intersect1d(ch_labels[0],channels_to_remove,return_indices=True)
    idx_final_eeg = np.delete(idx_final_eeg,idx)


    # # processing events ____________________________________________________________________________________________________________
    add_RunDay_toEvents(events, labels)


    # # processing eeg  _______________________________________________________________________________________________________________
    if applyLaplacian:  eeg = eeg @ lap['lapmask']
    eeg_lap_new = eeg[:,idx_final_eeg]    

    eeg_bands = sgnpr.get_bandranges(eeg_lap_new, bandranges, fs, filter_order,  filtType='bandpass')
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
    # eogVector = np.full((nwindows), np.nan)
    dayVector = np.full((nwindows), np.nan) 
    # modalityVector = np.full((nwindows), np.nan)
    protocolVector = np.full((nwindows), np.nan)

    idx_shift = 0
    for k in tqdm (np.unique(labels['Rk']), bar_format='{l_bar}{bar:40}{r_bar}'):
        idx_run = np.where(labels['Rk']==k)[0]
        t_events = events[events.run==k].reset_index(drop=True)
        t_events.loc[:,'pos'] = t_events.pos.values -(np.min(idx_run) if k>1 else 0)
        [t_covs,t_cov_events] = sgnpr.get_trNorm_covariance_matrix(eeg_bands[:,idx_run], t_events, windowsLength, windowsShift, fs, dispProgress=False)
        covs[:,idx_shift:idx_shift+t_covs.shape[1],:,:] = t_covs

        t_cov_events.pos = t_cov_events.pos + idx_shift
        cov_events = pd.concat([cov_events, t_cov_events], axis=0, ignore_index=True)

        runVector[idx_shift:idx_shift+t_covs.shape[1]] = k
        # eogVector[idx_shift:idx_shift+t_covs.shape[1]] = t_cov_events.loc[0,'eog']
        dayVector[idx_shift:idx_shift+t_covs.shape[1]] = t_cov_events.loc[0,'day']
        # modalityVector[idx_shift:idx_shift+t_covs.shape[1]] = labels['Mk'][idx_run[0]]
        protocolVector[idx_shift:idx_shift+t_covs.shape[1]] = labels['Pk'][idx_run[0]]
        
        idx_shift += t_covs.shape[1]

    cov_events = cov_events.astype(int)
    utilsVector = {'run': runVector, 'day': dayVector, 'protocol': protocolVector}

    if saveData:
        if saveName!='': filename = saveName
        else:
            if doLogBandPower:
                filename = filename.replace('dataset_user', 'covs_logBandPower_user')
            else:
                filename = filename.replace('user', 'covs_user')
        print('Saving data into ' + f'{pathData}{filename} ...')
        savemat(f'{pathData}{filename}',{'covs':covs,'cov_events':cov_events.to_numpy(), 'column_names': cov_events.columns.values, 'utilsVector': utilsVector, 'days':days})
    else:
        print('Data not saved')





def concatenate_cov_datasets(covs, cov_events, runs_labels, utilsVector, validRuns, days, classes, newDataDict):
    if len(cov_events)>0:
        print(f'Concatenating datasets: original dataset with {covs.shape[1]} samples, {len(cov_events)} events, {len(runs_labels)} runs, {len(utilsVector["run"])} utilsVector entries, {len(days)} days')
        if covs.ndim == 3:  covs = np.expand_dims(covs, axis=0)
    else:
        print(f'Concatenating datasets: original dataset is empty. Adapting the new dataset.')

    copy_utilsVector = utilsVector.copy()
    n_days = np.max(cov_events['day']) if len(cov_events)>0 else 0
    n_total_runs = np.max(cov_events['run']) if len(cov_events)>0 else 0
    n_samples = covs.shape[1] if len(covs)>0 else 0

    t_covs = newDataDict['covs']
    if t_covs.ndim == 3:    t_covs = np.expand_dims(t_covs, axis=0)
    if 'column_names' in newDataDict.keys():
        columns_name = [x[0] for x in newDataDict['column_names'][0]]
        t_cov_events = pd.DataFrame(newDataDict['cov_events'], columns=columns_name)
        t_utilsVector = dtmn.fix_mat(newDataDict['utilsVector'][0])
    else:
        t_cov_events = newDataDict['cov_events']
        t_utilsVector = newDataDict['utilsVector']
        n_samples = 1

    if isinstance(t_cov_events['pos'].iloc[0], np.ndarray):
        # print(t_cov_events.dtypes)
        getElements = lambda x: np.squeeze(x)
        t_cov_events = t_cov_events.applymap(getElements)
        print('Elements in cov_events converted from array to int or float')
    t_cov_events = t_cov_events.astype(int)
    # print(t_cov_events.dtypes)

    t_cov_events['rac'] = np.zeros(len(t_cov_events.typ)) if 'rac' not in t_cov_events.columns else t_cov_events['rac']
    t_cov_events['eog'] = np.zeros(len(t_cov_events.typ)) if 'eog' not in t_cov_events.columns else t_cov_events['eog']
    t_cov_events['ply'] = np.zeros(len(t_cov_events.typ)) if 'ply' not in t_cov_events.columns else t_cov_events['ply']

    t_isCFeedback = np.zeros(t_covs.shape[1])
    t_labels = np.full((t_covs.shape[1]), np.nan)
    for idx,ev in t_cov_events.loc[t_cov_events.typ==781,:].iterrows():
        t_isCFeedback[int(ev.pos):int(ev.pos+ev.dur-1)] = 1
        cues = t_cov_events.loc[:idx-1, :].loc[t_cov_events['typ'].isin(classes)]
        cue = cues['typ'].iloc[-1]
        t_labels[int(ev.pos):int(ev.pos + ev.dur - 1)] = cue

    t_runs_labels = np.empty((np.max(t_cov_events.run), 1), dtype=object)
    for nR in np.unique(t_cov_events.run):
        lbl = t_labels[t_utilsVector['run']==nR]
        t_runs_labels[nR-1] = np.empty((1), dtype=object)
        t_runs_labels[nR-1][0] = np.squeeze(np.array([lbl[~np.isnan(lbl)]]))
        t_runs_labels[nR-1][0]=lbl[~np.isnan(lbl)]
    t_runs_labels = t_runs_labels[:nR]

    t_cov_events.loc[:,'pos'] = t_cov_events.pos.values -1 + n_samples
    t_cov_events.loc[:,'day'] = t_cov_events.day.values + n_days
    t_cov_events.loc[:,'run'] = t_cov_events.run.values + n_total_runs

    if 'days' in newDataDict.keys():
        t_days = newDataDict['days']
    else:
        t_days = np.array(['03/10/2023','11/10/2024','14/10/2024','15/10/2024','18/10/2024','21/10/2024','22/10/2024'])

    t_utilsVector['eog'] = np.zeros(len(t_utilsVector['run']))
    t_utilsVector['day'] = t_utilsVector['day'] + n_days
    t_utilsVector['run'] = t_utilsVector['run'] + n_total_runs
    t_utilsVector['isCFeedback'] = t_isCFeedback

    if len(copy_utilsVector)>0:
        copy_utilsVector.pop('daysLabel') if 'daysLabel' in copy_utilsVector.keys() else None
        copy_utilsVector.pop('modality') if 'modality' in copy_utilsVector.keys() else None
        for key in copy_utilsVector.keys():
            copy_utilsVector[key] = np.concatenate((copy_utilsVector[key], t_utilsVector[key]))
    else:
        copy_utilsVector = t_utilsVector

    days = np.concatenate((days, t_days)) if len(days)>0 else t_days
    cov_events = pd.concat([cov_events, t_cov_events], axis=0, ignore_index=True) if len(cov_events)>0 else t_cov_events
    covs = np.concatenate((covs,t_covs), axis=1) if len(covs)>0 else t_covs
    runs_labels = np.concatenate((runs_labels, t_runs_labels), axis=0) if len(runs_labels)>0 else t_runs_labels
    validRuns = np.concatenate((validRuns, np.unique(t_cov_events.run))) if len(validRuns)>0 else np.unique(t_cov_events.run)

    return covs, cov_events, runs_labels, copy_utilsVector, validRuns, days


def add_RunDay_toEvents(events, labels):
    events['run'] = np.zeros(len(events.typ))
    for n in np.unique(labels['Rk']):
        pos_start_run = np.where(labels['Rk'] == n)[0][0]
        idx_events = np.where(events['pos']>=pos_start_run)[0][0]
        events.loc[idx_events:, 'run'] = n
    events['day'] = np.zeros(len(events.typ))
    for n in np.unique(labels['Dk']):
        pos_start_run = np.where(labels['Dk'] == n)[0][0]
        idx_events = np.where(events['pos']>=pos_start_run)[0][0]
        events.loc[idx_events:, 'day'] = n




def processDataset_PSD_user20232024(dataset, internalWindowsLength, windowsShift, pshift, mlength, fs=512):
    print('- Processing PSD dataset')
    # # # loading dataset ______________________________________________________________________________________________________________
    # if isinstance(pathData, str):
    #     file_path = f'{pathData}dataset_user_20192020.mat'
    #     dataset = loadmat(file_path)
    # else:
    #     dataset = pathData

    print('- Dataset loaded')

    # dataset = dataset['data']
    eeg = dataset['eeg']
    ev = dtmn.fix_mat(dataset['events'] )      
    labels = dtmn.fix_mat(dataset['procLabels'])

    # # processing events ____________________________________________________________________________________________________________
    events = pd.DataFrame(ev)
    events.columns = events.columns.str.lower()

    for idx in events.index:
        events.loc[idx,'run'] = labels['Rk'][int(events.loc[idx,'pos'])]
        events.loc[idx,'day'] = labels['Dk'][int(events.loc[idx,'pos'])]


    path = '/home/palatella/workspace/cap_utils/laplacian16.mat'
        
    lap = loadmat(path)
    laplacian_mask = lap['lapmask']


    # # processing eeg  _______________________________________________________________________________________________________________
    print('- Applying laplacian')
    eeg = eeg @ laplacian_mask



    ## load classifiers  _____________________________________________________________________________________________
    channelBandDict = {}
    models_path = '/home/palatella/workspace/f1_classifiers/'

    files = sorted([f for f in os.listdir(models_path) if ('.smr.mat' in f and ('2023' in f or '2024' in f))])
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
    dayVector = np.full((nwindows), np.nan) 
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
        dayVector[idx_shift:idx_shift+t_psd.shape[0]] = t_events.loc[0,'day']
        protocolVector[idx_shift:idx_shift+t_psd.shape[0]] = labels['Pk'][idx_run[0]]
        
        idx_shift += t_psd.shape[0]

    psd = np.log(psd)

    return psd, freq[np.isin(freq, frequencies)], psd_events, runVector, dayVector, protocolVector, channelBandDict

