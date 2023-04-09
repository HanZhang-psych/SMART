# -*- coding: utf-8 -*-
"""
@author: Jonathan van Leeuwen, 2018
"""
#==============================================================================
# This script runs the data anaysis for Dataset 1
#==============================================================================
import time
t = time.time()
print 'Starting Analysis...\n'
import pandas as pd
import numpy as np
import scipy.stats as st
from joblib import Parallel, delayed
np.set_printoptions(suppress = True)
# Reset the random function (for determenistic results)
np.random.seed(1234567891)

# Get the location of the data and load data
print 'Reading data...\n'
fileN = 'Dataset-1.p'
AllData = pd.read_pickle(fileN)
resultsFN = 'ResultsDataset-1.p'
permFN = 'PermDataset-1.p'

# Various settings
njobs = 4
nBins = [3,4,5,6,7]
sigmas = [10,20,30,40,50]

gMin = 100
gMax = 500
runPermutations = True
nPerms = 1000

# Make new X vector to project smoothed data to
newX = np.linspace(int(gMin),int(gMax),int(gMax) - int(gMin)+1)

# Extract meta information
FN = np.unique(AllData.subject_nr)

#==============================================================================
#==============================================================================
# # Functions used to do math and such
#==============================================================================
#==============================================================================
def weighArraysByColumn(arr,weights):
    '''
    '''
    # normalize weights
    weightsN = weights/np.sum(weights, axis = 0)
    # Get normalized values (add columns to get the weighted average)
    arrW = arr * weightsN
    return arrW

def perctilStat(dataSort, dataToBin, statCalc = 'mean', nrBins = 10):
    '''Needs np arrays as input for both data variables'''
    # Sort data
    data        = np.column_stack((dataSort, dataToBin))
    data        = data[data[:,0].argsort()]
    indexList   = range(0,len(data))

    # actually do the calculation
    binStat, bin_edges, binIndex = st.binned_statistic(indexList,\
                                                        data[:,1],\
                                                        statistic=statCalc,\
                                                        bins = nrBins)
    return binStat

def perctilStatEdges(dataSort, dataToBin, statCalc = 'mean', nrBins = 10):
    '''Needs np arrays as input for both data variables'''
    # Sort data
    data        = np.column_stack((dataSort, dataToBin))
    data        = data[data[:,0].argsort()]
    indexList   = range(0,len(data))

    # actually do the calculation
    binStat, bin_edges, binIndex = st.binned_statistic(indexList,\
                                                        data[:,1],\
                                                        statistic=statCalc,\
                                                        bins = nrBins)
    return bin_edges

def getHardBins(dataSort, dataToBin, bEdges, statCalc):
    '''
    statCalc can be either "mean" or "count"
    '''
    # Itterate over each bin
    res = np.zeros(len(bEdges)-1)
    for bNr in range(len(bEdges)-1):
        binBools = np.logical_and(dataSort > bEdges[bNr], dataSort <= bEdges[bNr+1])
        if statCalc == 'mean':
            res[bNr] = np.mean(dataToBin[binBools])
        elif statCalc == 'count':
            res[bNr] = np.sum(binBools)
    return res

def gaussFilt(x, y, newX, sigma):
    delta_x = newX[:, None] - x
    weights = np.exp(-delta_x*delta_x / (2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    dataWeight = np.sum(weights, axis= 1)
    weights /= np.sum(weights, axis=1, keepdims=True)
    y_eval = np.dot(weights, y)
    return y_eval, dataWeight

def getKDE(x, newX, sigma):
    """ 
    """
    # Get counts for each unique value and return ints
    unqX, countsX = np.unique(x, return_counts=True)
    unqX = np.array(unqX, dtype=int)
    
    # Make the vector on which we want to make the KDE
    x = np.arange(int(x.min()-50), int(x.max()+50))
    y = np.zeros(x.shape)
    indices = np.where(np.in1d(x, unqX))[0]
    # Make new vector with the counts
    y[np.array(indices, dtype=int)] = countsX

    # Run the KDE
    delta_x = newX[:, None] - x
    weights = np.exp(-delta_x*delta_x / (2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.dot(weights, y), unqX, countsX

#==============================================================================
# Functions used for secondary analysis (permutations)
#==============================================================================
# This version is a very fast version to do permutations against 0
def gausFiltPermZero(x, y, newX, sigma, nPerms):
    delta_x = newX[:, None] - x
    weights = np.exp(-delta_x*delta_x / (2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    dataWeight = np.sum(weights, axis= 1)
    weights /= np.sum(weights, axis=1, keepdims=True)
    y_eval = np.zeros((nPerms, len(newX)))
    y = y - np.average(y)
    for i in range(nPerms): 
        np.random.shuffle(y)
        y_eval[i,:] = np.dot(weights, y)
    return y_eval, dataWeight

def gausFiltPermCond(x1, y1, x2, y2, newX, sigma, nPerms):  
    # Merge vectors
    data = np.hstack((x1, x2))
    time = np.hstack((y1, y2))
    idx = np.arange(len(time))

    #Prealocate data storage
    pData1 = np.zeros((nPerms,len(newX)))
    pData2 = np.copy(pData1)
    pWeights1 = np.copy(pData1)
    pWeights2 = np.copy(pData1)       
    
    # Run the permutation testing   
    for perm in xrange(nPerms):
        # Here we shuffle the conditions
        np.random.shuffle(idx)
        # Split the indexes 
        pIdx1 = idx[0:len(x1)]
        pIdx2 = idx[len(x1):]
        
        # Extract permutated data 
        # Cond1
        pX1 = data[pIdx1]
        pY1 = time[pIdx1]
        #Cond2
        pX2 = data[pIdx2]
        pY2 = time[pIdx2]
        
        ####
        # Do the actual gaussian smoothing
        # Make gaussians Cond1
        pData1[perm,:], pWeights1[perm,:] = gaussFilt(pX1, pY1, newX, sigma)
        # Make gaussians Cond2
        pData2[perm,:], pWeights2[perm,:] = gaussFilt(pX2, pY2, newX, sigma)
        
    return pData1, pWeights1, pData2, pWeights2

#==============================================================================
# Main analysis function
#==============================================================================
def analyse(fn):
    #==========================================================================
    # Start with data handling
    #==========================================================================
    # Split data set into chunk for the current dataset
    data = AllData.loc[AllData.subject_nr == fn,:]
    data.reset_index(inplace = True)
    
    # Extract relevant data vectors (and store them )
    results = pd.DataFrame(data = [fn], columns = ['ppNr'])
    perm = results.copy()
    for key in data.keys():
        results[key] = [data[key].values]
    
    # Extract condition bools
    tMatchBool = results.match.values[0] == 'targ_match'
    nMatchBool = results.match.values[0] == 'none_match'
    dMatchBool = results.match.values[0] == 'distr_match'
    
    # Extract the original binns
    results['tMatchOrigT'] = [[results.saccade_latency.values[0][tMatchBool][results.bin.values[0][tMatchBool] == i+1].mean() for i in range(4)]]
    results['nMatchOrigT'] = [[results.saccade_latency.values[0][nMatchBool][results.bin.values[0][nMatchBool] == i+1].mean() for i in range(4)]]
    results['dMatchOrigT'] = [[results.saccade_latency.values[0][dMatchBool][results.bin.values[0][dMatchBool] == i+1].mean() for i in range(4)]]
    results['tMatchOrigC'] = [[results.saccade_correctness.values[0][tMatchBool][results.bin.values[0][tMatchBool] == i+1].mean() for i in range(4)]]
    results['nMatchOrigC'] = [[results.saccade_correctness.values[0][nMatchBool][results.bin.values[0][nMatchBool] == i+1].mean() for i in range(4)]]
    results['dMatchOrigC'] = [[results.saccade_correctness.values[0][dMatchBool][results.bin.values[0][dMatchBool] == i+1].mean() for i in range(4)]]
        
    # Extract timevector
    tMatchT = results.saccade_latency.values[0][tMatchBool]
    nMatchT = results.saccade_latency.values[0][nMatchBool]
    dMatchT = results.saccade_latency.values[0][dMatchBool]
    # Extract correct  vector
    tMatchC = results.saccade_correctness.values[0][tMatchBool]
    nMatchC = results.saccade_correctness.values[0][nMatchBool]
    dMatchC = results.saccade_correctness.values[0][dMatchBool]
    
    #==========================================================================
    # Do the within subjects binning
    #==========================================================================
    for b in nBins:
        # Get the curvature values
        tMatchCB = perctilStat(tMatchT, tMatchC, statCalc = 'mean', nrBins = b)
        nMatchCB = perctilStat(nMatchT, nMatchC, statCalc = 'mean', nrBins = b)
        dMatchCB = perctilStat(dMatchT, dMatchC, statCalc = 'mean', nrBins = b)

        # Get the time values
        tMatchTB = perctilStat(tMatchT, tMatchT, statCalc = 'mean', nrBins = b)
        nMatchTB = perctilStat(nMatchT, nMatchT, statCalc = 'mean', nrBins = b)
        dMatchTB = perctilStat(dMatchT, dMatchT, statCalc = 'mean', nrBins = b)

        
        # Store the results in dataframe
        results['tMatchCBins'+str(b)] = [tMatchCB]
        results['nMatchCBins'+str(b)] = [nMatchCB]
        results['dMatchCBins'+str(b)] = [dMatchCB]
        
        results['tMatchTBins'+str(b)] = [tMatchTB]
        results['nMatchTBins'+str(b)] =[nMatchTB]
        results['dMatchTBins'+str(b)] =[dMatchTB]
        
    #==========================================================================
    # Do the binning based on hard limits
    #==========================================================================
    binRange = np.arange((gMax-gMin)+1)
    for b in nBins:
        # Get the binCutoffs
        bEdges = perctilStatEdges(binRange,binRange, statCalc = 'mean', nrBins = b)+gMin
        # Get the curvature values
        tMatchCB = getHardBins(tMatchT, tMatchC, bEdges, 'mean')
        nMatchCB = getHardBins(nMatchT, nMatchC, bEdges, 'mean')
        dMatchCB = getHardBins(dMatchT, dMatchC, bEdges, 'mean')

        # Get the time values
        tMatchTB = getHardBins(tMatchT, tMatchT, bEdges, 'mean')
        nMatchTB = getHardBins(nMatchT, nMatchT, bEdges, 'mean')
        dMatchTB = getHardBins(dMatchT, dMatchT, bEdges, 'mean')
        
        # Store the results in dataframe
        results['tMatchCHardBins'+str(b)] = [tMatchCB]
        results['nMatchCHardBins'+str(b)] = [nMatchCB]
        results['dMatchCHardBins'+str(b)] = [dMatchCB]
        
        results['tMatchTHardBins'+str(b)] = [tMatchTB]
        results['nMatchTHardBins'+str(b)] =[nMatchTB]
        results['dMatchTHardBins'+str(b)] =[dMatchTB]
        
    #==========================================================================
    # Gaussian smoothing analysis
    #==========================================================================
    for sig in sigmas:        
        # Make gaussians 
        gaussTMatch = gaussFilt(tMatchT, tMatchC, newX, sig)
        gaussNMatch = gaussFilt(nMatchT, nMatchC, newX, sig)
        gaussDMatch = gaussFilt(dMatchT, dMatchC, newX, sig)

        # Stor the smoothed data
        results['gaussTMatch'+str(sig)] = [gaussTMatch[0]]
        results['gaussTMatchW'+str(sig)] = [gaussTMatch[1]]
        results['gaussNMatch'+str(sig)] = [gaussNMatch[0]]
        results['gaussNMatchW'+str(sig)] = [gaussNMatch[1]]
        results['gaussDMatch'+str(sig)] = [gaussDMatch[0]]
        results['gaussDMatchW'+str(sig)] = [gaussDMatch[1]]

    results['nTrials'] = np.sum([len(tMatchT),len(tMatchT), len(tMatchT)])
    results['allTimesT'] = [tMatchT]
    results['allTimesD'] = [dMatchT]
    
    #==========================================================================
    # Guassian permutation between conditions
    #==========================================================================    
    if runPermutations == True:
        for sig in sigmas: 
            sig = int(sig)
            # Run permutations
            tMatchPC, tMatchPWC, dMatchPC, dMatchPWC = gausFiltPermCond(tMatchT, tMatchC, dMatchT, dMatchC, newX, sig, nPerms)
            perm['tMatchPermC'+str(sig)] = [tMatchPC]
            perm['tMatchPermWC'+str(sig)] = [tMatchPWC]
            perm['dMatchPermC'+str(sig)] = [dMatchPC]
            perm['dMatchPermWC'+str(sig)] = [dMatchPWC]
            
    return results, perm

#==============================================================================
# Here we run the analysis in parallel for speed 
#==============================================================================
if __name__ == "__main__":
    print 'Itterating over participants in data set...\n'
    if runPermutations == True:
        print '\nRunning permutations, this might take a while!!\n\n'
    if len(FN) > 1 and runPermutations == True:
        AllResults = Parallel(n_jobs=njobs,verbose=9)(delayed(analyse)(fn = fn) for fn in FN)
    else:
        AllResults = [analyse(FN[i-1]) for i in FN]
        
    # Untangle results and perm from AllResults
    results = [i[0] for i in AllResults]
    results = pd.concat(results, ignore_index=True)
    perm = [i[1] for i in AllResults]
    perm = pd.concat(perm, ignore_index=True)
    
    # Calculate the KDE for each sigma
    allTimesT = np.hstack(results['allTimesT'])
    allTimesD = np.hstack(results['allTimesD'])
    
    for s in sigmas:
        # Target match KDE
        KDE, unqX, countsX = getKDE(allTimesT, newX, s)
        results['tKDE'+str(s)] = [KDE]*len(FN)
        results['tKDE_x'+str(s)] = [unqX]*len(FN)
        results['tKDE_x_counts'+str(s)] = [countsX]*len(FN)
        
        # Distractor match KDE
        KDE, unqX, countsX = getKDE(allTimesD, newX, s)
        results['dKDE'+str(s)] = [KDE]*len(FN)
        results['dKDE_x'+str(s)] = [unqX]*len(FN)
        results['dKDE_x_counts'+str(s)] = [countsX]*len(FN)
        
    # Save the results and permutations 
    print '\nSaving results:'
    print resultsFN
    results.to_pickle(resultsFN)
    if runPermutations == True:
        print '\nSaving permutations:'
        print permFN
        perm.to_pickle(permFN)

    print '\nDone!'
    print 'Duration:', (time.time() - t),'s\n'
    

