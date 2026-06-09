import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# # importing libraries ____________________________________________________________________________________________________________


import riemann_utils.covariances as rutils 
from pyriemann.utils.base import invsqrtm
import numpy as np
from tqdm import tqdm
import riemann_utils.plots as rplt
from pyriemann.utils.tangentspace import tangent_space
import matplotlib.pyplot as plt
from scipy.io import savemat
import riemann_utils.matrix_functions as mrtrf
from pyriemann.utils.distance import distance_riemann



def extract_run_centroids(covs, cov_events, validRuns, classes, runVector, isCFeedbackVector, runs_labels, runRef=None, saveData=False, pathData='', doLogBandPower=False, saveName=None):
    n_acc_run = validRuns.shape[0]
    if runRef is None:  runRef = validRuns[0]
    if runRef not in validRuns:
        raise ValueError('runRef not in validRuns')

    run_centroids = np.full((covs.shape[0], n_acc_run, len(classes), covs.shape[-2], covs.shape[-1]), np.nan)
    std_centroids = np.full((covs.shape[0], n_acc_run, len(classes)), np.nan)
    mAbsDev_centroids = np.full((covs.shape[0], n_acc_run, len(classes)), np.nan)
    std_eyeDistance = np.full((covs.shape[0], n_acc_run, len(classes)), np.nan)

    reshaped_eye = np.tile(np.eye(run_centroids.shape[-1]), (np.prod(run_centroids.shape[0:-2]), 1, 1)).reshape(run_centroids.shape)

    for count_run,nR in tqdm (enumerate(validRuns), total=n_acc_run, bar_format='{l_bar}{bar:40}{r_bar}'):
        # print('count: ' + str(count_run), '  nR: ' + str(nR), end="  ")
        chosen_idx = (runVector==nR) & (isCFeedbackVector)
        lbl = runs_labels[nR-1][0]

        data = covs[:,chosen_idx]
        flag = np.full((len(classes)), False)
        for idx, clss in enumerate(classes):
            if np.sum(lbl==clss) > 0:
                flag[idx] = True
                # print(str(clss), end="  ")
                mean_cov, _ = rutils.get_riemann_mean_covariance(data[:,lbl==clss], lbl[lbl==clss], show_progess=False, print_print=False)  
                run_centroids[:, count_run, idx] = mean_cov
                std_centroids[:, count_run, idx] = mrtrf.matrix_std(data[:,lbl==clss], mean_cov)
                mAbsDev_centroids[:, count_run, idx] = mrtrf.matrix_meanAbsoluteDeviation(data[:,lbl==clss], mean_cov)
                std_eyeDistance[:, count_run, idx] = mrtrf.matrix_std(data[:,lbl==clss], np.eye(data.shape[-1]))

        if not all(flag):
            raise ValueError('Missing classes')
        
    d_ToEye = distance_riemann(run_centroids, reshaped_eye)    


    first_run_events = cov_events.loc[cov_events.run==runRef,:]
    first_run_isFeedback = np.full(len(isCFeedbackVector), False)
    for idx, ev in first_run_events.loc[first_run_events.typ==781,:].iterrows():
        first_run_isFeedback[int(ev.pos):int(ev.pos+ev.dur-1)] = True
    data = covs[:,first_run_isFeedback]
    ref_angle, _ = rutils.get_riemann_mean_covariance(data, show_progess=False, print_print=False) 

    if saveData:
        if saveName is None:    saveName = 'run_centroids_user.mat' if not doLogBandPower else 'run_centroids_logBandPower_user.mat'
        print('Saving data into ' + f'{pathData}{saveName} ...')
        savemat(f'{pathData}{saveName}',{'run_centroids':run_centroids, 'ref_angle':ref_angle, 'd_ToEye':d_ToEye,
                                        'std_centroids':std_centroids, 'std_eyeDistance':std_eyeDistance, 'mAbsDev_centroids':mAbsDev_centroids})
    else:
        print('Run centroids not saved')


def extract_day_centroids(covs, cov_events, validRuns, classes, runVector, dayVector, isCFeedbackVector, runs_labels, dayRef=None, saveData=False, pathData='', doLogBandPower=False, saveName=None):
    runMask = np.isin(runVector, validRuns)
    validDays = np.unique(dayVector[runMask])
    n_days = len(validDays)

    if dayRef is None:  dayRef = validDays[0]
    if dayRef not in validDays:
        raise ValueError('dayRef not in validDays')

    day_centroids = np.full((covs.shape[0], n_days, len(classes), covs.shape[-2], covs.shape[-1]), np.nan)
    std_centroids = np.full((covs.shape[0], n_days, len(classes)), np.nan)
    mAbsDev_centroids = np.full((covs.shape[0], n_days, len(classes)), np.nan)
    std_eyeDistance = np.full((covs.shape[0], n_days, len(classes)), np.nan)

    reshaped_eye = np.tile(np.eye(day_centroids.shape[-1]), (np.prod(day_centroids.shape[0:-2]), 1, 1)).reshape(day_centroids.shape)

    for count_day,nD in tqdm (enumerate(validDays), total=n_days, bar_format='{l_bar}{bar:40}{r_bar}'):
        # print('count: ' + str(count_day), '  nD: ' + str(nD), end="  ")
        chosen_idx = (dayVector==nD) & (isCFeedbackVector) & (runMask)
        selected_runs = np.unique(runVector[chosen_idx]).astype(int)
        if not np.isin(selected_runs, validRuns).all():
            raise ValueError('Some runs in the day are not in validRuns')
        lbl = np.concatenate(runs_labels[selected_runs-1,0])

        data = covs[:,chosen_idx]
        flag = np.full((len(classes)), False)
        for idx, clss in enumerate(classes):
            if np.sum(lbl==clss) > 0:
                flag[idx] = True
                # print(str(clss), end="  ")
                mean_cov, _ = rutils.get_riemann_mean_covariance(data[:,lbl==clss], lbl[lbl==clss], show_progess=False, print_print=False)  
                day_centroids[:, count_day, idx] = mean_cov
                std_centroids[:, count_day, idx] = mrtrf.matrix_std(data[:,lbl==clss], mean_cov)
                mAbsDev_centroids[:, count_day, idx] = mrtrf.matrix_meanAbsoluteDeviation(data[:,lbl==clss], mean_cov)
                std_eyeDistance[:, count_day, idx] = mrtrf.matrix_std(data[:,lbl==clss], np.eye(data.shape[-1]))

        if not all(flag):
            raise ValueError('Missing classes')
        
    d_ToEye = distance_riemann(day_centroids, reshaped_eye)    


    first_run_events = cov_events.loc[cov_events.day==dayRef+1,:]  # days are 1-indexed in the events
    first_run_isFeedback = np.full(len(isCFeedbackVector), False)
    for idx, ev in first_run_events.loc[first_run_events.typ==781,:].iterrows():
        if not ev.run in validRuns: continue
        first_run_isFeedback[int(ev.pos):int(ev.pos+ev.dur-1)] = True
    data = covs[:,first_run_isFeedback]
    ref_angle, _ = rutils.get_riemann_mean_covariance(data, show_progess=False, print_print=False) 

    if saveData:
        if saveName is None:    saveName = 'day_centroids_user.mat' if not doLogBandPower else 'day_centroids_logBandPower_user.mat'
        print('Saving data into ' + f'{pathData}{saveName} ...')
        savemat(f'{pathData}{saveName}',{'day_centroids':day_centroids, 'ref_angle':ref_angle, 'd_ToEye':d_ToEye,
                                        'std_centroids':std_centroids, 'std_eyeDistance':std_eyeDistance, 'mAbsDev_centroids':mAbsDev_centroids})
    else:
        print('Day centroids not saved')




def center_covariances(covs, dayVector, isCFeedbackVector, referenceDay=0, mean_cov=None, inv_sqrt_mean_cov=None, saveTransformMatrices=False, pathData='', saveName=None):
    idx_reference = (dayVector==referenceDay) & (isCFeedbackVector)
    data = covs[:,idx_reference]

    if mean_cov is None:    mean_cov, _ = rutils.get_riemann_mean_covariance(data, show_progess=False, print_print=False)
    if inv_sqrt_mean_cov is None: inv_sqrt_mean_cov = np.expand_dims(invsqrtm(mean_cov),1)
    covs_firstDay_centered= rutils.center_covariances(covs, mean_cov, inv_sqrt_mean_cov)

    if saveTransformMatrices:
        savemat(f'{pathData}{saveName}',{'mean_cov':mean_cov, 'inv_sqrt_mean_cov':inv_sqrt_mean_cov})

    return covs_firstDay_centered, mean_cov, inv_sqrt_mean_cov











def plot_centoids(covs, cov_events, labelVector):
    n_samples = covs.shape[1]
    runVector = rutils.get_CueVector_onFeedback(cov_events, n_samples, column_name='run')
    data = covs[:,runVector==0]
    lbl = labelVector[runVector==0]

    mean_cov_first, _ = rutils.get_riemann_mean_covariance(data, lbl, show_progess=False, print_print=False)
    inv_sqrt_mean_cov_first = np.expand_dims(invsqrtm(mean_cov_first),1)
    covs_firstrun_centered= rutils.center_covariances(covs, mean_cov_first, inv_sqrt_mean_cov_first)

    n_runs = np.unique(cov_events.run.values[~np.isnan(cov_events.run.values)])
    centroids = np.full((covs.shape[0], 2*len(n_runs), covs.shape[-2], covs.shape[-1]), np.nan)
    run_vector = np.full((2*len(n_runs)),np.nan)
    lbl_vector = np.full((2*len(n_runs)),np.nan)

    for nR in tqdm (range(len(n_runs)), bar_format='{l_bar}{bar:40}{r_bar}'):
        run_idx = (runVector==nR) & (~np.isnan(labelVector))
        data = covs_firstrun_centered[:,run_idx]
        lbl = labelVector[run_idx]
        for idx, clss in enumerate(np.unique(lbl)):
            mean_cov, _ = rutils.get_riemann_mean_covariance(data[:,lbl==clss], lbl[lbl==clss], show_progess=False, print_print=False)  
            centroids[:, 2*nR+idx] = mean_cov
            run_vector[2*nR+idx] = nR
            lbl_vector[2*nR+idx] = clss

    center_point = np.eye(centroids.shape[-1])
    refs = np.expand_dims(center_point , 0)
    refs = np.tile(refs, (centroids.shape[0], 1, 1, 1))
    data_centroids = np.concatenate((refs, centroids), axis=1)
    data_run_vector = np.concatenate(( np.full((1),np.nan), run_vector), axis=0)
    data_run_vector = np.squeeze(data_run_vector)

    n_neighbors = 250
    min_dist = 0.7

    tan_centroids  = tangent_space(data_centroids, center_point)
    pca_centroids,expl_var = rutils.get_nd_position(tan_centroids, n_components=3)
    umap_centroids,_ = rplt.umap_reduction(tan_centroids, n_components=3, min_dist=min_dist, n_neighbors=n_neighbors)

    n_cmp = range(1,tan_centroids.shape[-1]+1)
    variance = np.zeros((tan_centroids.shape[-1],2))
    for i in n_cmp:
        _,variance[i-1] = rutils.get_nd_position(tan_centroids, n_components=i, suppress_output=True)
    plt.plot(n_cmp, variance)

    idx = range(0, pca_centroids.shape[1], 2)
    idx1 = range(1, pca_centroids.shape[1], 2)
    rplt.plot_cartesian(pca_centroids[1], labels=data_run_vector, sessionVector=lbl_vector, fig_size=(17,7), trialsGradientVector=data_run_vector, point_size=20)

    pca_centroids2d = rutils.get_nd_position(tan_centroids, n_components=2)
    rplt.plot_cartesian(pca_centroids2d[1], labels=data_run_vector, sessionVector=lbl_vector, fig_size=(17,7), trialsGradientVector=data_run_vector, point_size=20)

    rplt.plot_cartesian(umap_centroids[0], labels=data_run_vector, sessionVector=lbl_vector, fig_size=(17,7), trialsGradientVector=data_run_vector, point_size=20)

    umap_centroids2d,_ = rplt.umap_reduction(tan_centroids, n_components=2, min_dist=min_dist, n_neighbors=n_neighbors)
    rplt.plot_cartesian(umap_centroids2d[1], labels=data_run_vector, sessionVector=lbl_vector, fig_size=(17,7), trialsGradientVector=data_run_vector, point_size=20)

