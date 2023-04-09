# -*- coding: utf-8 -*-
"""
@author: Jonathan van Leeuwen, 2018
"""
#==============================================================================
# This script plots the results for Dataset 2
#==============================================================================
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.stats as st
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot

np.random.seed(1234567890)
np.set_printoptions(suppress = True)
#plt.close('all')

# Get the location of the data and load data
resultsFN = 'ResultsDataset-2.p' # 
results = pd.read_pickle(resultsFN)

# Load permutations
permFN = 'PermDataset-2.p' #
perm = pd.read_pickle(permFN)

# Constants
nBins = [3,4,5,6,7]
sigmas = [10,20,30,40,50] 
nPerms = len(perm.dispPerm1Z10[0][1])

# Various settings
lineWidth = 2
displCol = [130,23,25]
noDisplCol = [0,255,255]
Oppacity = 0.25
xMin = 100
xMax = 300
yMin = -3.25
yMax = 3.25
sOffset = 0.75
markerW = 2.0
capsize = 3 # size of the cap of the errorbars
ms = 10
xLabelSize = 12
yLabelSize = 25
titelSize = 15
tickSize = 13
legendSize = 13

# Transform colors
displCol = [i/255.0 for i in displCol]
noDisplCol = [i/255.0 for i in noDisplCol]

# Set tick Size
plt.rcParams['xtick.labelsize'] = tickSize 
plt.rcParams['ytick.labelsize'] = tickSize            
#==============================================================================
# usefull functions
#==============================================================================
def maximizeFigure():
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

def nanSTD(data):
    notNan = np.invert(np.isnan(data))
    stdv = np.zeros(data.shape[1])
    for i in range(len(stdv)):
        stdv[i] = np.std(data[:,i][notNan[:,i]])
    return stdv
    
def weighArraysByColumn(arr,weights):
    '''
    '''
    # normalize weights
    weightsN = weights/np.sum(weights, axis = 0)
    # Get normalized values (add columns to get the weighted average)
    arrW = arr * weightsN
    return arrW

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
    
#==============================================================================
# Functions for running statistics
#==============================================================================
def ConfOneSample95(data):
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

    '''
    stdv        = np.std(data, axis = 0)
    SE          = stdv/np.sqrt(len(data))
    confInt     = scipy.stats.t.ppf(0.975,len(data)-1) * SE

    return confInt

def weighConfOneSample95(cond, weights):
    u'''
    '''
    # Number of non nan values per time point
    N = len(cond) - np.sum(np.isnan(cond), axis = 0) 
    
    # get weighted average
    weighData = weighArraysByColumn(cond, weights)
    weighAv = np.nansum(weighData, axis=0)
    
    # Do the rest    
    sum_of_weights = np.nansum(weights,axis=0)
    nominator = np.nansum(weights**2 * ((cond - weighAv)**2),axis=0)
    sem = np.sqrt((nominator*N)/((N-1)*sum_of_weights**2))
    t_val = scipy.stats.t.ppf(0.975,N-1)
    confInt95 = t_val * sem
    return confInt95

def weighted_ttest(cond, weights, baseline=0):
    u'''
    This function
    Returns:
    '''
    # Normalize weights per columns
    normW = weights/np.nansum(weights, axis = 0)
    weighAv = np.nansum(normW*cond, axis = 0)

    # AVB weighted SEM
    N = len(cond) - np.sum(np.isnan(cond), axis = 0)
        
    sum_of_weights = np.nansum(weights,axis=0)
    nominator = (np.nansum(weights**2 * ((cond - weighAv)**2),axis=0))/(N-1)
    sem = np.sqrt(N*nominator/(sum_of_weights**2))

    tvals = (weighAv-baseline)/sem
    pvals = scipy.stats.t.sf(np.abs(tvals), N-1)*2

    return tvals, pvals

def weighted_ttest_rel(cond1, cond2, weights1, weights2):
    '''
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
# Cluster statistics against baseline
#==============================================================================
def clusterStat_oneSamp(cond, weights, baseline, sigThresh=0.05):
    '''
    One sampled t-test for each column of cond1 and baseline
    Returns all significant clusters and sum of t-values for the clusters
    T-values are the sum of the absolute t-values in a cluster
    '''
    tValues, pValues = weighted_ttest(cond, weights, baseline)

    sigArr = pValues < sigThresh
    cl, clInd= getCluster(sigArr)
    sigCl = [clInd[i] for i in range(len(cl)) if cl[i][0] == True]
    sumTvals = [np.sum(np.abs(tValues[i])) for i in sigCl]
    return sigCl, sumTvals

def permuteClusterStat(perm1, perm2, permWeights1, permWeights2, sigThresh=0.05):
    '''
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
    displ = np.vstack(results['displCBins'+str(b)].values)
    displT = np.vstack(results['displTBins'+str(b)].values)
    
    ###
    # Average
    avDisp = np.nanmean(displ,axis = 0)    
    avDispT = np.nanmean(displT,axis = 0)    
    
    ### 
    # Calculate statistics
    tVal, pVal = scipy.stats.ttest_rel(displ, np.zeros(displ.shape))    
    pValBon = pVal*len(pVal)
    sigVec = pVal < 0.05
    sigVecBon = pValBon < 0.05
    
    ###
    # Variations
    confInt95 = ConfOneSample95(displ)
    stdDisplT = nanSTD(displT)
    
    ###
    # plot data
    cAx = ax[idx,0]
    cAx.plot(avDispT, avDisp, '^:', linewidth = lineWidth, color = displCol, markeredgecolor = [0,0,0])
    
    # Plot errorbars
    cAx.errorbar(avDispT, avDisp, confInt95, stdDisplT, fmt='^', color = displCol, capsize = capsize, markeredgecolor = [0,0,0])
  
    # Plot statistics
    for i in range(len(sigVec)):
        # Plot uncorrected values
        if sigVec[i] == True and not sigVecBon[i]:
            yPos = np.max(avDisp[i])+confInt95[i]+sOffset
            xPos = avDispT[i]
            if yPos >= yMax:
                yPos = yMax-0.5
            #cAx.plot(xPos,yPos, 'k+', ms = 5, mew=markerW)
        # Plot corrected
        elif sigVecBon[i] == True:
            yPos = np.max(avDisp[i])+confInt95[i]+sOffset
            xPos = avDispT[i]
            if yPos >= yMax:
                yPos = yMax-0.5
            cAx.plot(xPos,yPos, 'k*', ms = ms)
            print '\nBins', b
            print 'SigVinBin', xPos
    ###
    # Format plot
    cAx.plot([xMin,xMax], [0, 0], 'k--' )
    cAx.set_xlim(xMin, xMax)
    cAx.set_ylim(yMin, yMax)
    cAx.set_yticks([-2,0,2])
    if idx != len(nBins)-1:
        cAx.xaxis.set_ticklabels([])
    else:
        cAx.set_xlabel('Intersaccadic interval (ms)', fontsize=xLabelSize)
    if idx ==0:
        cAx.set_title('Vincentizing bins', fontsize=titelSize)
    cAx.legend([b], fontsize=legendSize, markerscale=0, scatterpoints=None, handlelength=0, frameon=False, loc='upper right')

    # Set aspect ratio
    changeAspectRatio(cAx)
    
#==============================================================================
# Plot hard binned data 
#==============================================================================
print '\n\nHard binned results'
for idx, b in enumerate(nBins):
    displ = np.vstack(results['displCHardBins'+str(b)].values)
    displT = np.vstack(results['displTHardBins'+str(b)].values)
 
    ###
    # Average
    avDisp = np.nanmean(displ,axis = 0)    
    avDispT = np.nanmean(displT,axis = 0)    

    ### 
    # Calculate statistics
    tVal, pVal = scipy.stats.ttest_rel(displ, np.zeros(displ.shape), nan_policy = 'omit')
    pValBon = pVal*len(pVal)
    sigVec = pVal < 0.05
    sigVecBon = pValBon < 0.05
    
    ###
    # Variations
    confInt95 = ConfOneSample95(displ)
    stdDisplT = nanSTD(displT)
    
    ###
    # plot data
    cAx = ax[idx,1]
    cAx.plot(avDispT, avDisp, 's-.', linewidth = lineWidth, color = displCol, markeredgecolor = [0,0,0])
  
    # Plot errorbars
    cAx.errorbar(avDispT, avDisp, confInt95, stdDisplT, fmt='s', color = displCol,capsize = capsize, markeredgecolor = [0,0,0])

    # Plot statistics
    for i in range(len(sigVec)):
        # Plot uncorrected values
        if sigVec[i] == True and not sigVecBon[i]:
            yPos = np.max(avDisp[i])+confInt95[i]+sOffset
            xPos = avDispT[i]
            if yPos >= yMax:
                yPos = yMax-0.5
            #cAx.plot(xPos,yPos, 'k+', ms = 5, mew=markerW)
        # Plot corrected
        elif sigVecBon[i] == True:
            yPos = np.max(avDisp[i])+confInt95[i]+sOffset
            xPos = avDispT[i]
            if yPos >= yMax:
                yPos = yMax-0.5
            cAx.plot(xPos,yPos, 'k*', ms = ms)
            print '\nBins', b
            print 'SigHardBin', xPos

            
    ###
    # Format plot
    cAx.plot([xMin,xMax], [0, 0], 'k--' )
    cAx.set_xlim(xMin, xMax)
    cAx.set_ylim(yMin, yMax)
    cAx.yaxis.set_ticklabels([])
    if idx != len(nBins)-1:
        cAx.xaxis.set_ticklabels([])
    else:
        cAx.set_xlabel('Intersaccadic interval (ms)', fontsize=xLabelSize)
    if idx ==0:
        cAx.set_title('Hard limit bins', fontsize=titelSize)
    cAx.legend([b], fontsize=legendSize, markerscale=0, handletextpad=0.0, handlelength=0, frameon=False, loc='upper right')

    # Set aspect ratio
    changeAspectRatio(cAx)
    
#==============================================================================
# Plot Smoothed data
#==============================================================================
print '\n\nGaussian smoothed results'
All95ints = np.zeros(len(sigmas))
for idx, s in enumerate(reversed(sigmas)):
    xVect = np.arange(xMin, xMax+1)
    displ = np.vstack(results['gausDisp'+str(s)])
    displW = np.vstack(results['gausDispW'+str(s)])

    # Permutations, Reshaped (dim1 = pp, dim2 = Time, dim3 = perm )
    nPoints = len(xVect)
    nPerm = len(perm['dispPerm1Z'+str(s)][0][1])
    dispPermZ1 = np.zeros((len(perm), nPoints, nPerm))
    dispPermZW1 = dispPermZ1.copy()
    dispPermZ2 = dispPermZ1.copy()
    dispPermZW2 = dispPermZ1.copy()
    for pp in range(len(perm)):
        dispPermZ1[pp,:,:] = perm['dispPerm1Z'+str(s)][pp]
        dispPermZW1[pp,:,:] = perm['dispPerm1ZW'+str(s)][pp]
        dispPermZ2[pp,:,:] = perm['dispPerm2Z'+str(s)][pp]
        dispPermZW2[pp,:,:] = perm['dispPerm2ZW'+str(s)][pp]
              
    ###
    # Average
    weighDispl = weighArraysByColumn(displ, displW)
    avDispl = np.sum(weighDispl, axis = 0)

    ### 
    # Calculate statistics (weighted)
    sigCL, sumTvals = clusterStat_oneSamp(displ, displW, 0, sigThresh=0.05)

    ###
    # Calculate permutation distributions
    permDistr = permuteClusterStat(dispPermZ1, dispPermZ2, dispPermZW1, dispPermZW2, 0.05)
    perm95 = np.percentile(permDistr, 95)
    All95ints[idx] = perm95
    
    ###
    # Variations
    conf95 = weighConfOneSample95(displ, displW)
    
    ###
    # Get axis
    changeAspectRatio(ax[idx,2])
    ax[idx,2].axis('off')
    cAx = host_subplot(len(nBins),4,(idx*4)+3)
    
    ###
    # plot data
    cAx.plot(xVect, avDispl, '-', linewidth = lineWidth, color = displCol)

    # Plot significant time points
    for ind, i in enumerate(sigCL):
        cAx.plot(xVect[i], avDispl[i], 'k-', lineWidth = lineWidth*1.5)
        # Plot asterix for signficant clusters
        if sumTvals[ind] >= perm95:
            print '\nsigma', s
            print 'startCluster', xVect[i][0]
            print 'endCluster', xVect[i][-1]
            xPos = np.average(xVect[i])
            yPos = np.max(avDispl[int(np.mean(i))])+sOffset+conf95[int(np.mean(i))] 
            if yPos >= yMax:
                yPos = yMax-0.5
            cAx.plot(xPos,yPos, 'k*', ms = ms)
        
    # Plot errorbars
    cAx.fill_between(xVect, avDispl - conf95, avDispl + conf95, color = displCol, alpha = 0.25)

    # Format plot
    cAx.plot([xMin,xMax], [0, 0], 'k--' )
    cAx.set_xlim(xMin, xMax)
    cAx.set_ylim(yMin, yMax)
    cAx.yaxis.set_ticklabels([])
    if idx != len(nBins)-1:
        cAx.xaxis.set_ticklabels([])
    else:
        cAx.set_xlabel('Intersaccadic interval (ms)', fontsize=xLabelSize)
    if idx ==0:
        cAx.set_title('SMART method', fontsize=titelSize)

    # Set aspect ratio
    changeAspectRatio(cAx)
    
    ###
    # Plot KDE    
    KDE = results['KDE'+str(s)][0]
    unqT = results['KDE_x'+str(s)][0]
    countT = results['KDE_x_counts'+str(s)][0]
    maxT = np.max(KDE)*8
    axT = cAx.twinx()
    axT.plot(xVect, KDE, '--k', alpha = 0.3)
    axT.bar(unqT, countT, color='k', alpha = 0.3)  
    axT.set_yticks(np.linspace(0,np.max(KDE),2, dtype=int))
    axT.set_ylim(0, maxT)
    axT.set_xlim(xMin, xMax)
        
    ###
    # Plot permutation distributions
    pAx = ax[idx,3]
    pAx.hist(permDistr, 100, log=True, bottom=0.8)
    yLims =pAx.get_ylim()
    pAx.plot([perm95,perm95], yLims, color = 'r', linewidth = 2, label = '95th percentile')
    for tV in sumTvals:
        pAx.plot([tV, tV], yLims, color = 'k', linewidth = 2)
        
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
ax1.set_ylabel("Curvature (deg of arc)", size=yLabelSize)

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
