# -*- coding: utf-8 -*-
"""
@author: Jonathan van Leeuwen, 2018
"""
#==============================================================================
# This script runs the data anaysis for Dataset 2
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
fileN = 'Dataset-2.p'
AllData = pd.read_pickle(fileN)
resultsFN = 'ResultsDataset-2.p'
permFN = 'PermDataset-2.p'

# Various settings
njobs = 4
nBins = [3,4,5,6,7]
sigmas = [10,20,30,40,50]

gMin = 100
gMax = 300
runPermutations = True
nPerms = 1000

# Extract meta information
FN = np.unique(AllData.ppNr)

# Make new X vector to project smoothed data to
newX = np.linspace(int(gMin),int(gMax),int(gMax) - int(gMin)+1)

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
def heavyside(arr, cutOff = 0.5):
    """ 
    """
    res = np.zeros(arr.shape)
    res[arr >= cutOff] = 1
    return res

def makeBaseline(x, baseline=0.0, binary=False):
    """ 
    """
    base = np.random.normal(baseline, np.std(x), x.shape)
    if binary:
        base = heavyside(base, 0.5)
    return base

def permute(x1, y1, x2=None, y2=None, newX=[None], sigma=20, nPerms=1000, baseline=None):
    ''' 
    '''
    # Input handeling
    if newX[0]==None:
        newX = np.arange(100,501)
    if baseline!=None:
        x2=x1.copy() 
        y2=np.zeros(y1.shape)+baseline
        
    # Merge vectors
    data = np.hstack((x1, x2))
    depVar = np.hstack((y1, y2))
    idx = np.arange(len(depVar))

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
        pY1 = depVar[pIdx1]
        #Cond2
        pX2 = data[pIdx2]
        pY2 = depVar[pIdx2]

        ####
        # Do the actual gaussian smoothing
        # Make gaussians Cond1
        pData1[perm,:], pWeights1[perm,:] = gaussFilt(pX1, pY1, newX, sigma)
        # Make gaussians Cond2
        pData2[perm,:], pWeights2[perm,:] = gaussFilt(pX2, pY2, newX, sigma)

    return pData1.T, pWeights1.T, pData2.T, pWeights2.T

#==============================================================================
# Main analysis function
#==============================================================================
def analyse(fn):
    #==========================================================================
    # Start with data handling
    #==========================================================================
    # Split data set into chunk for the current dataset
    data = AllData.loc[AllData.ppNr == fn,:]
    data.reset_index(inplace = True)
    
    # Extract relevant data vectors
    wDisp = data.Sacc2JumpSameToOppositeCurvTrials.values[0]
    aDisp = data.Sacc2JumpOppsiteToSameCurvTrials.values[0]

    wDispT = data.Sacc2JumpSameToOppositeCurvTrialsInterSaccInt.values[0]
    aDispT = data.Sacc2JumpOppsiteToSameCurvTrialsInterSaccInt.values[0]
    
    # Initiate pd dataframe
    results = pd.DataFrame(data = [fn], columns = ['ppNr'])
    perm = results.copy()
    
    #==========================================================================
    # Do the within subjects binning
    #==========================================================================
    for b in nBins:
        # Get the curvature values
        wDispCurv = perctilStat(wDispT, wDisp, statCalc = 'mean', nrBins = b)
        aDispCurv = perctilStat(aDispT, aDisp, statCalc = 'mean', nrBins = b)
        
        # Get the time values
        wDispTB = perctilStat(wDispT, wDispT, statCalc = 'mean', nrBins = b)
        aDispTB = perctilStat(aDispT, aDispT, statCalc = 'mean', nrBins = b)
        
        # Get curvature values for each bin
        displCurv = wDispCurv - aDispCurv
        # Get average time for each bin
        dispAvT = np.average([wDispTB, aDispTB], axis = 0)

        # Store the results in dataframe
        results['displCBins'+str(b)] = [displCurv]
        results['displTBins'+str(b)] = [dispAvT]
    
    #==========================================================================
    # Do the binning based on hard limits
    #==========================================================================
    binRange = np.arange((gMax-gMin)+1)
    for b in nBins:
        # Get the binCutoffs
        bEdges = perctilStatEdges(binRange,binRange, statCalc = 'mean', nrBins = b)+gMin
        # Get the curvature values
        wDispCurv = getHardBins(wDispT, wDisp, bEdges, 'mean')
        aDispCurv = getHardBins(aDispT, aDisp, bEdges, 'mean')
        # Get the time values
        wDispTB = getHardBins(wDispT, wDispT, bEdges, 'mean')
        aDispTB = getHardBins(aDispT, aDispT, bEdges, 'mean')
        # Get curvature values for each bin
        displCurv = wDispCurv - aDispCurv
        # Get average time for each bin
        dispAvT = np.average([wDispTB, aDispTB], axis = 0)
        # Store the results in dataframe
        results['displCHardBins'+str(b)] = [displCurv]
        results['displTHardBins'+str(b)] = [dispAvT]
        
    #==========================================================================
    # Gaussian smoothing analysis
    #==========================================================================
    for sig in sigmas:        
        # Make gaussians 
        gausWDisp = gaussFilt(wDispT, wDisp, newX, sig)
        gausAgDisp = gaussFilt(aDispT, aDisp, newX, sig)
        # Stor the smoothed data
        results['gausDisp'+str(sig)] = [gausWDisp[0] - gausAgDisp[0]]
        results['gausDispW'+str(sig)] = [gausWDisp[1] + gausAgDisp[1]]
    results['nTrials'] = np.sum([len(wDispT), len(aDispT)])
    results['allTimes'] = [np.hstack([wDispT, aDispT])]
    
    #==========================================================================
    # Guassian permutation against 0 (Z = Zero) for each condition individually
    #==========================================================================
    if runPermutations == True:
        for sig in sigmas: 
            sig = int(sig)
            # Run permutation            
            wDispPZ1, wDispPWZ1, wDispPZ2, wDispPWZ2 = permute(
                                                            wDispT, 
                                                            wDisp, 
                                                            newX=newX, 
                                                            sigma=sig, 
                                                            nPerms=nPerms, 
                                                            baseline=0,
                                                            )
            agDispPZ1, agDispPWZ1, agDispPZ2, agDispPWZ2 = permute(
                                                            aDispT, 
                                                            aDisp, 
                                                            newX=newX, 
                                                            sigma=sig, 
                                                            nPerms=nPerms, 
                                                            baseline=0,
                                                            )
                
            # Get the permutated curvatures for displacement and no displacement 
            perm['dispPerm1Z'+str(sig)] = [wDispPZ1 - agDispPZ1]
            perm['dispPerm1ZW'+str(sig)] = [wDispPWZ1 + agDispPWZ1]
            perm['dispPerm2Z'+str(sig)] = [wDispPZ2 - agDispPZ2]
            perm['dispPerm2ZW'+str(sig)] = [wDispPWZ2 + agDispPWZ2]
            
    return results, perm

#==============================================================================
# Here we run the analysis in parallel for speed 
#==============================================================================
if __name__ == "__main__":
    print 'Itterating over participants in data set...\n'
    if len(FN) > 1 and runPermutations == True:
        print '\nRrunning permutations, this might take a while!!'
        AllResults = Parallel(n_jobs=njobs,verbose=9)(delayed(analyse)(fn = fn) for fn in FN)
    else:
        AllResults = [analyse(FN[i-1]) for i in FN]
        
    # Untangle results and perm from AllResults
    results = [i[0] for i in AllResults]
    results = pd.concat(results, ignore_index=True)
    perm = [i[1] for i in AllResults]
    perm = pd.concat(perm, ignore_index=True)
    
    # Calculate the KDE for each sigma
    allTimes = np.hstack(results['allTimes'])
    for s in sigmas:
        KDE, unqX, countsX = getKDE(allTimes, newX, s)
        results['KDE'+str(s)] = [KDE]*len(FN)
        results['KDE_x'+str(s)] = [unqX]*len(FN)
        results['KDE_x_counts'+str(s)] = [countsX]*len(FN)
        
    # Save the results and permutations 
    print '\nSaving results:'
    print resultsFN
    results.to_pickle(resultsFN)
    if runPermutations == True:
        print '\nSaving permutations:'
        print permFN
        perm.to_pickle(permFN)
        del perm

    print '\nDone!'
    print 'Duration:', (time.time() - t),'s\n'
    









