# -*- coding: utf-8 -*-
"""
Utility functions for running SMART analysis of eye-tracking data

@author: Han Zhang
"""
import pandas as pd
import numpy as np
import time
from SMARTClass_HZ import SMART

def run_smart_oneSamp(data, item, pt, params, runPerm=True):
    t = time.time()
    oneSamp = SMART(data, item, pt)
    oneSamp.runSmooth(params['krnSize'], params['minTime'], params['maxTime'], params['stepSize'])
    if runPerm:
        oneSamp.runPermutations(params['nPerm'], params['baseline'], params['nJobs'])
        oneSamp.runStats(params['sigLevel'])

    # remove some attribtutes that take huge memory
    cols = ['permData1','permWeight1','permData2', 'permWeight2']
    for name in cols:
        delattr(oneSamp, name)    
    print(f'time taken: {time.time() - t} seconds')

    return oneSamp

def run_smart_pairedSamp(data, item1, pt1, item2, pt2, params, runPerm=True):
    t = time.time()
    pairedSamp = SMART(data, item1, pt1, item2, pt2)
    pairedSamp.runSmooth(params['krnSize'], params['minTime'], params['maxTime'], params['stepSize'])
    if runPerm:
        pairedSamp.runPermutations(params['nPerm'], nJobs=params['nJobs'])
        pairedSamp.runStats(params['sigLevel'])
        
    # remove some attribtutes that take huge memory
    cols = ['permData1','permWeight1','permData2', 'permWeight2']
    for name in cols:
        delattr(pairedSamp, name)  

    print(f'time taken: {time.time() - t} seconds')
    return pairedSamp

# peak
def get_peak(oneSamp, peak=True):
    l = []
    if peak:
        v = max(oneSamp.weighDv1Average)
    else:
        v = min(oneSamp.weighDv1Average)
    for x, y in enumerate(oneSamp.weighDv1Average):
        if y == v:
            l.append((v, oneSamp.timeVect[x]))
    return l

# find accuracy given t
def get_acc(oneSamp, t):
    index = np.where(oneSamp.timeVect == t)[0][0]
    return oneSamp.weighDv1Average[index]

# get significant clusters
def get_SMART_results(smartResDict, paired=False):
    if paired:
        res = {'key':[], 'index': [], 'baseline':[], 'sigLevel':[], 'sumTvals':[], 'sigThres':[], 'permDistLen':[], 'start': [], 'end': [], 'start_value_0': [], 'end_value_0': [], 'start_value_1': [], 'end_value_1': []}
    else:
        res = {'key':[], 'index': [],  'baseline':[], 'sigLevel':[], 'sumTvals':[], 'sigThres':[], 'permDistLen':[], 'start': [], 'end': [], 'start_value': [], 'end_value': []}

    for key, smartRes in smartResDict.items():
        for ind, cluster in enumerate(smartRes.sigCL):
            if smartRes.sumTvals[ind] >= smartRes.sigThres:
                start_index = cluster[0]
                end_index = cluster[-1]
                res['key'].append(key)
                res['index'].append(ind)
                res['baseline'].append(smartRes.baseline)
                res['sigLevel'].append(smartRes.sigLevel)
                res['sumTvals'].append(smartRes.sumTvals[ind]) # The sum of tvalues for the each significant cluster
                res['sigThres'].append(smartRes.sigThres) # Cluster size threshold
                res['permDistLen'].append(len(smartRes.permDistr)) # useful for checking number of permutations returning NaNs
                res['start'].append(smartRes.timeVect[start_index])
                res['end'].append(smartRes.timeVect[end_index])
                if paired:
                    res['start_value_0'].append(smartRes.weighDv1Average[start_index])
                    res['end_value_0'].append(smartRes.weighDv1Average[end_index])
                    res['start_value_1'].append(smartRes.weighDv2Average[start_index])
                    res['end_value_1'].append(smartRes.weighDv2Average[end_index])
                else:
                    res['start_value'].append(smartRes.weighDv1Average[start_index])
                    res['end_value'].append(smartRes.weighDv1Average[end_index])

    return pd.DataFrame(res)

# def assign_object(trial, screen_center = (0,0), start_thresh = 80, end_thresh = [80, 500], item_dev_thresh=200, center_positions=[[0,-300],[300,0],[-300,0],[0,300]]):
#     # define vars
#     land_pos = 'None'
#     item_dev = np.nan

#     # extract info from row    
#     x_c, y_c = screen_center
#     x0 = trial.start_x
#     y0 = trial.start_y
#     x = trial.end_x
#     y = trial.end_y
#     target_pos = trial.target_pos
#     distractor = trial.distractor_cond
#     distractor_pos = trial.distractor_pos

#     # calculate dist to screen center
#     d2center0 = np.sqrt((x0 - x_c) ** 2 + (y0 - y_c) ** 2)
#     d2center = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
#     if (d2center0 < start_thresh) & (d2center > end_thresh[0]) & (d2center < end_thresh[1]):

#         # get closest item to saccade end position
#         positions = np.array(center_positions)
#         distances = np.sqrt((x - positions[:, 0])**2 + (y - positions[:, 1])**2)
#         obj_pos = center_positions[np.argmin(distances)]
#         obj_dist = np.min(distances)

#         # if the sac end position is not too far away from the item
#         if obj_dist < item_dev_thresh:
#             if np.all(obj_pos == target_pos):  # on target
#                 land_pos= 'target'
#                 item_dev = obj_dist
#             elif distractor == 'P' and np.all(obj_pos == distractor_pos):
#                 land_pos= 'distractor'
#                 item_dev = obj_dist
#             else:
#                 land_pos= 'other'
#                 item_dev = obj_dist

#     return (land_pos, item_dev)