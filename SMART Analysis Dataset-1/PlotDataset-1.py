# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:39:30 2017

@author: Jonathan van Leeuwen, 2018
"""
#==============================================================================
# This script plots the results for Dataset 1
#==============================================================================
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.stats
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot

np.random.seed(1234567890)
np.set_printoptions(suppress = True)
plt.close('all')

# Get the location of the data and load data
resultsFN = 'ResultsDataset-1.p'
results = pd.read_pickle(resultsFN)

# Load permutations
permFN = 'PermDataset-1.p'
perm = pd.read_pickle(permFN)

# Constants
nBins = [3,4,5,6,7]
sigmas = [10,20,30,40,50]
nPerms = len(perm.tMatchPermC10[0])

# Various settings
lineWidth = 2
capsize = 3 # size of the cap of the errorbars
tMatchCol = [130,23,25]
nMatchCol = [0,255,255]
dMatchCol = [0,255,255]
Oppacity = 0.25
xMin = 100
xMax = 500
yMin = -0.5
yMax = 1.5
sOffset = 0.2
markerW = 2
ms = 10
xLabelSize = 12
yLabelSize = 25
titelSize = 15
tickSize = 13
legendSize = 13

# Transform colors
tMatchCol = [i/255.0 for i in tMatchCol]
nMatchCol = [i/255.0 for i in nMatchCol]
dMatchCol = [i/255.0 for i in dMatchCol]

# Set tick Size
plt.rcParams['xtick.labelsize'] = tickSize 
plt.rcParams['ytick.labelsize'] = tickSize  
            
#==============================================================================
# usefull functions
#==============================================================================
def changeAspectRatio(ax,asp=(4,3)):
    '''
    ax = The axis to change 
    asp = The aspect ratio
        e.g. if ratio 4:3, asp = (4,3)
    '''
    yLimMin, yLimMax = ax.get_ylim()
    xLimMin, xLimMax = ax.get_xlim()
    xR = xLimMax-xLimMin
    yR = yLimMax-yLimMin
    ax.set_aspect(xR/yR * (asp[1]/float(asp[0])))
    
def maximizeFigure():
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

def nanSTD(data):
    notNan = np.invert(np.isnan(data))
    stdv = np.zeros(data.shape[1])
    for i in range(len(stdv)):
        stdv[i] = np.std(data[:,i][notNan[:,i]])
    return stdv
    
def withinSubConf(data):
    u'''
    This function returns the 95% confidence interval for a data set
    Data from each participant ordered in its own row and conditions
    in columns.
    The errorbars are calculated as follows:
        new value = old value – participant average + grand average
        Then get the stdv over all the new values,
        Then calculate (SE = Std.dev. / sqrt(6))
        (= 1.96 * SE; size of 95% CI error bars)

    Returns:
        +- 95 confidenceinterval for each column (one value for each column in data)

    It should be noted that:
        "However, there is some ambiguity in having error bars
        for anything that is not a single-factor-two-level design."
        
    Also note:
        This function removes any NaN values before calculation
    Calculation taken from: http://www.cogsci.nl/blog/tutorials/156-an-easy-way-to-create-graphs-with-within-subject-error-bars

    Cousineau, D. (2005). Confidence intervals in within-subject designs: A simpler solution to Loftus and Masson’s method. Tutorial in Quantitative Methods for Psychology, 1(1), 4–45.
    '''
    # remove nan values
    #nanVals = np.isnan(data[:,0])+np.isnan(data[:,1])+np.isnan(data[:,2])
    nanVals = np.isnan(data[:,0])+np.isnan(data[:,1])
    data = data[np.invert(nanVals)]
    
    grandAv     = np.nanmean(data)
    partAv      = np.transpose(np.nanmean(data, axis =1))
    partAv      = partAv[:,np.newaxis] # transpose the one dimensional vector
    newValues   = data - partAv + grandAv
    stdv        = np.std(newValues, axis = 0)
    SE          = stdv/np.sqrt(len(data[:,0]))
    confInt     = scipy.stats.t.ppf(0.975,len(data[:,0])-1) * SE

    return confInt

def weighArraysByColumn(arr,weights):
    '''
    '''
    # normalize weights
    weightsN = weights/np.nansum(weights, axis = 0)
    # Get normalized values (add columns to get the weighted average)
    arrW = arr * weightsN
    return arrW

#==============================================================================
# Functions for running statistics
#==============================================================================
def weighPairedConf95(cond1, cond2, weights1, weights2):
    u'''
     '''                               
    # Normalize weights per columns
    normW1 = weights1/np.nansum(weights1, axis = 0)
    normW2 = weights2/np.nansum(weights2, axis = 0)
    weighAv = np.nansum(normW1*cond1 - normW2*cond2, axis = 0)
    
    # Determine N
    N = len(cond1) - np.sum(np.isnan(cond1), axis = 0)
    N2 = len(cond2) - np.sum(np.isnan(cond2), axis = 0)
    if np.array(N).shape:
        N[N2<N] = N2[N2<N]
    else:
        N = np.min([N, N2])
        
    # Weighted SEM   
    nominator = (np.nansum(normW1*normW2 * (((cond1-cond2) - weighAv)**2),axis=0))/(N-1)
    sem = np.sqrt(N*nominator)
    t_val = scipy.stats.t.ppf(0.975,N-1)
    confInt = (t_val * sem)/2.
    
    return confInt

def weighted_ttest_rel(cond1, cond2, weights1, weights2):
    u'''
    This function
    Returns:
    '''
    # Normalize weights per columns
    normW1 = weights1/np.nansum(weights1, axis = 0)
    normW2 = weights2/np.nansum(weights2, axis = 0)
    weighAv = np.nansum(normW1*cond1 - normW2*cond2, axis = 0)

    # Determine N
    N = len(cond1) - np.sum(np.isnan(cond1), axis = 0)
    N2 = len(cond2) - np.sum(np.isnan(cond2), axis = 0)
    if np.array(N).shape:
        N[N2<N] = N2[N2<N]
    else:
        N = np.min([N, N2])
        
    # Weighted SEM  
    nominator = (np.nansum(normW1*normW2 * (((cond1-cond2) - weighAv)**2),axis=0))/(N-1)
    sem = np.sqrt(N*nominator)
      
    tvals = weighAv/sem
    pvals = scipy.stats.t.sf(np.abs(tvals), N-1)*2

    return tvals, pvals

#==============================================================================
# Get clusters
#==============================================================================
def getCluster(data):
    '''
    Splits data into clusters
    returns clusters and indexes
    '''
    clusters = np.split(data, np.where(np.diff(data) != 0)[0]+1)
    indx = np.split(np.arange(len(data)), np.where(np.diff(data) != 0)[0]+1)
    return clusters, indx


#==============================================================================
# Cluster statistics between conditions (Paired testing) 
#==============================================================================
def clusterStat_rel(cond1, cond2, weights1, weights2, sigThresh=0.05):
    '''
    Paired samples t-test for each column of cond1 and cond2
    Returns all significant clusters and sum of t-values for the clusters
    T-values are the sum of the absolute t-values in a cluster
    '''
    tValues, pValues = weighted_ttest_rel(cond1, cond2, weights1, weights2)

    sigArr = pValues < sigThresh
    cl, clInd= getCluster(sigArr)
    sigCl = [clInd[i] for i in range(len(cl)) if cl[i][0] == True]
    sumTvals = [np.sum(np.abs(tValues[i])) for i in sigCl]
    return sigCl, sumTvals

def clusterStatPerm_rel(perm1, perm2, permWeights1, permWeights2, sigThresh=0.05):
    '''
    Within subjects t-test for each column of cond1 and cond2
    Returns all significant clusters and sum of t-values for the clusters
    T-values are the sum of the absolute t-values in a cluster

    Assumes that the inputs are permutations in 3dimension
    pp in dim 1, test along dim2 and each permutation in dim 3
    '''
    permDistr = []
    tValues, pValues = weighted_ttest_rel(perm1, perm2, permWeights1, permWeights2)
    tValues = tValues.transpose()
    sigArr = (pValues<0.05).transpose()
    for indx in xrange(len(perm1[0,0,:])):
        cl, clInd= getCluster(sigArr[indx,:])
        sigCl = [clInd[i] for i in range(len(cl)) if cl[i][0] == True]
        if len(sigCl)  != 0:
            permDistr.append(np.max([np.sum(np.abs(tValues[indx,i])) for i in sigCl]))
        else:
            permDistr.append(np.max(tValues[indx,:]))
    return np.array(permDistr)

#==============================================================================
# Initiate figure
#==============================================================================
#f, ax = plt.subplots(len(nBins), 4, sharex = True, sharey = True)
f, ax = plt.subplots(len(nBins), 4, figsize=(16,12))
ax[0,0].axis([xMin, xMax, yMin, yMax])
#maximizeFigure()

#==============================================================================
# Plot within subjects binned data 
#==============================================================================
print '\n\nVincentized binned results'
for idx, b in enumerate(nBins):
    tMatch = np.vstack(results['tMatchCBins'+str(b)].values)
    tMatchT = np.vstack(results['tMatchTBins'+str(b)].values)
    dMatch = np.vstack(results['dMatchCBins'+str(b)].values)
    dMatchT = np.vstack(results['dMatchTBins'+str(b)].values)
    
    ###
    # Average
    avTMatch = np.nanmean(tMatch,axis = 0)    
    avTMatchT = np.nanmean(tMatchT,axis = 0)      
    avDMatch = np.nanmean(dMatch,axis = 0)    
    avDMatchT = np.nanmean(dMatchT,axis = 0)    
    
    ### 
    # Calculate statistics
    tVal, pVal = scipy.stats.ttest_rel(tMatch, dMatch)    
    pValBon = pVal*len(pVal)
    sigVec = pVal < 0.05
    sigVecBon = pValBon < 0.05
    
    ###
    # Variations
    confInt95 =  np.vstack([withinSubConf(np.vstack((tMatch[:,i], dMatch[:,i])).T) for i in range(b)])
    stdTMatchT = nanSTD(tMatchT)
    stdDMatchT = nanSTD(dMatchT)
    
    ###
    # plot data
    cAx = ax[idx,0]
    cAx.plot(avTMatchT, avTMatch, '^:', linewidth = lineWidth, color = tMatchCol, markeredgecolor = [0,0,0])
    cAx.plot(avDMatchT, avDMatch, '^:', linewidth = lineWidth, color = dMatchCol, markeredgecolor = [0,0,0])
    
    # Plot errorbars
    cAx.errorbar(avTMatchT, avTMatch, confInt95[:,0], stdTMatchT, fmt='^', color = tMatchCol, capsize = capsize, markeredgecolor = [0,0,0])
    cAx.errorbar(avDMatchT, avDMatch, confInt95[:,1], stdDMatchT, fmt='^', color = dMatchCol, capsize = capsize, markeredgecolor = [0,0,0])
    
    # Plot statistics
    for i in range(len(sigVec)):
        # Plot corrected
        if sigVecBon[i] == True:
            yPos = np.max([avTMatch[i], avDMatch[i]])+np.max(confInt95[i])+sOffset
            xPos = np.average([avTMatchT[i], avDMatchT[i]])
            if yPos >= yMax:
                yPos = yMax-0.5
            cAx.plot(xPos,yPos, 'k*', ms = ms)
        else:
            xPos = np.average([avTMatchT[i], avDMatchT[i]])
            print '\nBins', b
            print 'SigVinBin', xPos
    ###
    # Format plot
    cAx.plot([xMin,xMax], [0, 0], 'k--' )
    cAx.set_xlim(xMin, xMax)
    cAx.set_ylim(yMin, yMax)
    cAx.set_yticks([0,1])
    if idx != len(nBins)-1:
        cAx.xaxis.set_ticklabels([])
    else:
        cAx.set_xlabel('Saccade latency (ms)', fontsize=xLabelSize)
    if idx ==0:
        cAx.set_title('Vincentizing bins', fontsize=titelSize)
    cAx.legend([b], fontsize=legendSize, markerscale=0, handletextpad=0.0, handlelength=0, frameon=False, loc='upper right')

    # Set aspect ratio
    changeAspectRatio(cAx)
    
#==============================================================================
# Plot hard binned data 
#==============================================================================
print '\n\nHard binned results'
for idx, b in enumerate(nBins):
    tMatch = np.vstack(results['tMatchCHardBins'+str(b)].values)
    tMatchT = np.vstack(results['tMatchTHardBins'+str(b)].values)
    dMatch = np.vstack(results['dMatchCHardBins'+str(b)].values)
    dMatchT = np.vstack(results['dMatchTHardBins'+str(b)].values)
    
    ###
    # Average
    avTMatch = np.nanmean(tMatch,axis = 0)    
    avTMatchT = np.nanmean(tMatchT,axis = 0)    
    avDMatch = np.nanmean(dMatch,axis = 0)    
    avDMatchT = np.nanmean(dMatchT,axis = 0)  

    ### 
    # Calculate statistics
    tVal, pVal = scipy.stats.ttest_rel(tMatch,dMatch, nan_policy = 'omit')   
    pValBon = pVal*len(pVal)
    sigVec = pVal < 0.05
    sigVecBon = pValBon < 0.05
    
    ###
    # Variations
    confInt95 =  np.vstack([withinSubConf(np.vstack((tMatch[:,i], dMatch[:,i])).T) for i in range(b)])
    stdTMatchT = nanSTD(tMatchT)
    stdDMatchT = nanSTD(dMatchT)
    
    ###
    # plot data
    cAx = ax[idx,1]
    cAx.plot(avTMatchT, avTMatch, 's-.', linewidth = lineWidth, color = tMatchCol, markeredgecolor = [0,0,0])
    cAx.plot(avDMatchT, avDMatch, 's-.', linewidth = lineWidth, color = dMatchCol, markeredgecolor = [0,0,0])
    
    # Plot errorbars
    cAx.errorbar(avTMatchT, avTMatch, confInt95[:,0], stdTMatchT, fmt='s', color = tMatchCol, capsize = capsize, markeredgecolor = [0,0,0])
    cAx.errorbar(avDMatchT, avDMatch, confInt95[:,1], stdDMatchT, fmt='s', color = dMatchCol, capsize = capsize, markeredgecolor = [0,0,0])

    # Plot statistics
    for i in range(len(sigVec)):
        # Plot corrected
        if sigVecBon[i] == True:
            yPos = np.max([avTMatch[i], avDMatch[i]])+np.max(confInt95[i])+sOffset
            xPos = np.average([avTMatchT[i], avDMatchT[i]])
            if yPos >= yMax:
                yPos = yMax-0.5
            cAx.plot(xPos,yPos, 'k*', ms = ms)
        else:
            xPos = np.average([avTMatchT[i], avDMatchT[i]])
            print '\nBins', b
            print 'SigHardBin', xPos
            
    ###
    # Format plot
    cAx.plot([xMin,xMax], [0, 0], 'k--' )
    cAx.set_xlim(xMin, xMax)
    cAx.set_ylim(yMin, yMax)
    cAx.set_yticks([0,1])
    cAx.yaxis.set_ticklabels([])
    if idx != len(nBins)-1:
        cAx.xaxis.set_ticklabels([])
    else:
        cAx.set_xlabel('Saccade latency (ms)', fontsize=xLabelSize)
    if idx ==0:
        cAx.set_title('Hard limit bins', fontsize=titelSize)
    cAx.legend([b], fontsize=legendSize, markerscale=0, handletextpad=0.0, handlelength=0, frameon=False, loc='upper right')

    # Set aspect ratio
    changeAspectRatio(cAx)
    
#==============================================================================
# Plot Smoothed data
#==============================================================================
print '\n\nGaussian smoothed results'
for idx, s in enumerate(reversed(sigmas)):
    xVect = np.arange(xMin, xMax+1)
    tMatch = np.vstack(results['gaussTMatch'+str(s)])
    tMatchW = np.vstack(results['gaussTMatchW'+str(s)])
    dMatch = np.vstack(results['gaussDMatch'+str(s)])
    dMatchW = np.vstack(results['gaussDMatchW'+str(s)])
    
    # Permutations, Reshaped (dim1 = pp, dim2 = Time, dim3 = perm )
    nPoints = len(xVect)
    nPerm = len(perm['tMatchPermC'+str(s)][0])
    tMatchPermC = np.zeros((len(perm), nPoints, nPerm))
    tMatchPermCW = tMatchPermC.copy()
    dMatchPermC = tMatchPermC.copy()
    dMatchPermCW = tMatchPermC.copy()
    for pp in range(len(perm)):
        for p in range(nPerm):
            tMatchPermC[pp,:,p] = perm['tMatchPermC'+str(s)][pp][p]
            tMatchPermCW[pp,:,p] = perm['tMatchPermWC'+str(s)][pp][p]
            dMatchPermC[pp,:,p] = perm['dMatchPermC'+str(s)][pp][p]
            dMatchPermCW[pp,:,p] = perm['dMatchPermWC'+str(s)][pp][p]
    
    # Weighted permutations
    weighTMatchPerm = weighArraysByColumn(tMatchPermC, tMatchPermCW)
    weighDMatchPerm = weighArraysByColumn(dMatchPermC, dMatchPermCW)

    # Average (Ignore nan values)
    weighTMatch = weighArraysByColumn(tMatch, tMatchW)
    weighDMatch = weighArraysByColumn(dMatch, dMatchW)
    avTMatch = np.nansum(weighTMatch, axis = 0)
    avDMatch = np.nansum(weighDMatch, axis = 0)
    
    ### 
    # Calculate statistics (weighted)
    sigCL, sumTvals = clusterStat_rel(tMatch, dMatch, tMatchW, dMatchW, sigThresh=0.05)

    ###
    # Calculate permutation distributions
    permDistr = clusterStatPerm_rel(tMatchPermC, dMatchPermC, tMatchPermCW, dMatchPermCW, sigThresh=0.05)
    perm95 = np.percentile(permDistr, 95)
    
    ###
    # Variations
    conf95 = weighPairedConf95(tMatch, dMatch, tMatchW, dMatchW)

    ###
    # Get axis
    changeAspectRatio(ax[idx,2])
    ax[idx,2].axis('off')
    cAx = host_subplot(5,4,(idx*4)+3)
    
    ###
    # plot data
    cAx.plot(xVect, avTMatch, '-', linewidth = lineWidth, color = tMatchCol)
    cAx.plot(xVect, avDMatch, '-', linewidth = lineWidth, color = dMatchCol)
    
    # Plot significant time points
    for ind, i in enumerate(sigCL):
        cAx.plot(xVect[i], avTMatch[i], 'k-', lineWidth = lineWidth*1.5)
        cAx.plot(xVect[i], avDMatch[i], 'k-', lineWidth = lineWidth*1.5)
        # Plot asterix for signficant clusters
        if sumTvals[ind] >= perm95:
            xPos = np.average(xVect[i])
            yPos = np.max([avTMatch[int(np.mean(i))], avDMatch[int(np.mean(i))]])+sOffset+np.max(conf95[int(np.mean(i))])
            if yPos >= yMax:
                yPos = yMax-0.5
            cAx.plot(xPos,yPos, 'k*', ms = ms)
            print '\nsigma', s
            print 'startCluster', xVect[i][0]
            print 'endCluster', xVect[i][-1]
        
    # Plot errorbars
    cAx.fill_between(xVect, avTMatch - conf95, avTMatch + conf95, color = tMatchCol, alpha = 0.25)
    cAx.fill_between(xVect, avDMatch - conf95, avDMatch + conf95, color = dMatchCol, alpha = 0.25)
    
    # Format plot
    cAx.plot([xMin,xMax], [0, 0], 'k--' )
    cAx.set_xlim(xMin, xMax)
    cAx.set_ylim(yMin, yMax)
    cAx.set_yticks([0,1])
    cAx.yaxis.set_ticklabels([])
    if idx != len(nBins)-1:
        cAx.xaxis.set_ticklabels([])
    else:
        cAx.set_xlabel('Saccade latency (ms)', fontsize=xLabelSize)
    if idx ==0:
        cAx.set_title('SMART method', fontsize=titelSize)

    # Get KDE
    tKDE = results['tKDE'+str(s)][0]
    tunqT = results['tKDE_x'+str(s)][0]
    tcountT = results['tKDE_x_counts'+str(s)][0]
    dKDE = results['dKDE'+str(s)][0]
    dunqT = results['dKDE_x'+str(s)][0]
    dcountT = results['dKDE_x_counts'+str(s)][0]
    # Plot KDE
    maxT = np.max(np.hstack([tKDE, dKDE]))*8
    axT = cAx.twinx()
    axT.plot(xVect, tKDE, '--k', alpha = 0.6)
    axT.plot(xVect, dKDE, '-k', alpha = 0.3)
    axT.bar(tunqT, tcountT, color='k', alpha = 0.6)
    axT.bar(dunqT, dcountT, color='k', alpha = 0.3)
    axT.set_ylim(0, maxT)
    axT.set_yticks(np.linspace(0,np.max(np.hstack([tKDE, dKDE])),2, dtype=int))
    axT.set_ylim(0, maxT)
    axT.set_xlim(xMin, xMax)
        
    # Set aspect ratio
    changeAspectRatio(cAx)
    
    ###
    # Plot permutation distributions
    pAx = ax[idx,3]
    pAx.hist(permDistr, 100, log=True, bottom=0.8)
    pAx.plot([perm95,perm95], pAx.get_ylim(), color = 'r', linewidth = 2, label = '95th percentile')
    for tV in sumTvals:
        pAx.plot([tV, tV], pAx.get_ylim(), color = 'k', linewidth = 2)
        
    # Set xaxis
    pAx.yaxis.tick_right()
    pAx.yaxis.set_ticks_position('right')
    if idx == len(nBins)-1:
        pAx.set_xlabel('Sum of cluster t-values', fontsize=xLabelSize)
    if idx ==0:
        pAx.set_title('Permutation distribution', fontsize=titelSize)
    cAx.legend([s], fontsize=legendSize, markerscale=0, handletextpad=0.0, handlelength=0, frameon=False, loc='upper right')
    pAx.set_yscale('log')
    
    # Set aspect ratio
    changeAspectRatio(pAx)
    
#==============================================================================
# Do some figure formating
#==============================================================================
    plt.subplots_adjust(wspace=0)
    
# add a big axes, hide frame
ax1 = f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
ax1.set_yticks([])

# Set left ylabel
ax1.yaxis.set_label_coords(-0.03,0.5)
ax1.set_ylabel("Proportion correct saccades", size=yLabelSize)

# Set right ylabel
ax2 = ax1.twinx()
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
ax2.set_frame_on(False)
ax2.yaxis.set_label_position('right')
ax2.set_ylabel("Frequency (log)", size=yLabelSize)
ax2.set_yticks([])
ax2.yaxis.set_label_coords(1.05,0.5)


changeAspectRatio(ax1)

