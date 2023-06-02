#! /usr/bin/env python3

'''
jp52example

A short script showing an example of how the jp52mod class 'jp52base'
can be used to create a series of plots showing the group velocity
output curves.

Follow the code from beneath the " if __name__=='__main__' "
statement.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jp52mod import jp52base
from numpy import sqrt, arange

def pull_out_group_vel_params(jp52obj):
    '''
    Pull out some parameters of interest from the
    jp52model object, and save to a Pandas dataframe.
    '''
    df = pd.DataFrame(columns=['gvecMin','gvecMinFreq','branch3Freq'])

    #------------------------------------------------------------------
    # Identifying the group speed minimum, and corresponding frequency
    #------------------------------------------------------------------
    bI = min(jp52obj.gdict['gvec_I'])
    bII = min(jp52obj.gdict['gvec_II'])
    print('b1,b2:',bI,bII)
    group_speed_min = np.min([bI,bII])

    # Now need to link minimum val to corresponding frequency   
    indmin = np.where(jp52obj.gdict['gvec_I']==group_speed_min)[0]
    if len(indmin) == 0:
        # Then the minimum must be in branch II
        indmin = np.where(jp52obj.gdict['gvec_II']==group_speed_min)[0]
        freqmin = jp52obj.gdict['f_II'][indmin[0]]
    else:
        freqmin = jp52obj.gdict['f_I'][indmin[0]] # The frequency was in Branch I

    #-------------------------------------------------------------------
    # Identifying the group speeds at particular frequencies in Branch 1
    #-------------------------------------------------------------------


    if (np.max(jp52obj.gdict['f_I'])<1.0) or  (np.min(jp52obj.gdict['f_I'])>1.0):
        gvec1hz = np.nan
    else:
        ind1hz = np.argmin(np.abs(jp52obj.gdict['f_I']-1.0))
        gvec1hz = jp52obj.gdict['gvec_I'][ind1hz]

    if (np.max(jp52obj.gdict['f_I'])<1.25) or  (np.min(jp52obj.gdict['f_I'])>1.25):
        gvec1p25hz = np.nan
    else:        
        ind1p25hz = np.argmin(np.abs(jp52obj.gdict['f_I']-1.25))
        gvec1p25hz = jp52obj.gdict['gvec_I'][ind1p25hz]

    #-------------------------------------------------------------------
    # Populate the output dataframe
    #-------------------------------------------------------------------
    df.loc[0,'gvecMinFreq'] = freqmin # Frequency of Minimum group velocity from Branches I and II 
    df.loc[0,'gvecMin'] = group_speed_min # Minimum group velocity from Branches I and II 

    df.loc[0,'gvec1hz'] = gvec1hz # Minimum group velocity from Branches I and II 
    df.loc[0,'gvec1p25hz'] = gvec1p25hz # Minimum group velocity from Branches I and II 

    df.loc[0,'branch3Freq'] = np.median(jp52obj.gdict['f_III'])


    return df



if __name__=='__main__':

    # picdir - the directory into which the plots are saved. Will
    # need to be updated for whichever system the user is on.
    picdir = '/Users/pn22327/Documents/foulnessdata/code/JP52_modelling/temp_pics_4param'

    # Initialising a dataframe into which I'll put a few parameters
    # of interest:
    # gvecMin -> the minimum group velocity
    # gvecMinFreq -> the frequency at which the minimum group velocity occurs
    # branch3Freq -> the median frequency of the calculated branch 3
    df = pd.DataFrame(columns=['gvecMin','gvecMinFreq','branch3Freq', 'H', 'beta1', 'alpha1', 'beta2'])

    # Here I'm going to set up a base model, and then loop over
    # a number of changes in the shear wave velocity in the
    # top layer - similar loops can be constructed for other parameters.

    # Base Model
    alpha0 = 340. # Speed of sound in air (m/s)
    rho0 = 1.29 # Density of air (kg/m^3)
    alpha1 = 1500. # P-wave speed in upper solid layer (m/s)
    rho1 = 1800. # Density of upper solid layer (kg/m^3)
    beta2 = 1000. # Shear-wave speed in lower solid layer (m/s)
    alpha2 = 2700. # P-wave speed in lower solid layer (m/s)
    rho2 = 2000. # Density of lower solid layer (kg/m^3)
    H = 85. # Thickness of upper solid layer (m)


    # Loop over a set of representative shear-wave speeds for the upper
    # solid layer - note that these are below the acoustic velocity in
    # air.

    beta2vec = arange(1000., 1600., 100.)

    alpha1vec = arange(1600., 2000., 100.)

    Hvec = arange(70., 120., 10.)

    beta1vec = arange(230.,310., 10.)

    for beta2val in beta2vec:

        for alpha1val in alpha1vec:

            for Hval in Hvec:

                for beta1 in beta1vec:
                    paramdict = {'alpha0':alpha0,
                                'rho0': rho0,
                                'beta1':beta1,
                                'alpha1':alpha1val,
                                'rho1':rho1,
                                'beta2':beta2val,
                                'alpha2':alpha2,
                                'rho2':rho2,
                                'H':Hval}

                    # Sets up the model with the parameter dictionary
                    jp52test = jp52base(inputdict=paramdict)
                    # Returns the phase velocity curves (will be plotted if figfile is not None)
                    jp52test.return_phase_vel_curves(pltflag=False,asymptote=0.99) #,figfile=os.path.join(picdir,'test_model_b1_{}_Cvel.png'.format(beta1)))
                    # Returns the group velocity curves (non-dimensionalised)
                    # If pltflag = True, plots the curves to screen
                    jp52test.return_group_vel_curves(pltflag=False)
                    # Now need function to return a frequency vector and a group velocity vector in
                    # 'real' dimensions -> will require a layer thickness (H)
                    # If pltflag = True, plots the curves to screen
                    jp52test.gval_into_physical_units(pltflag=False)
                    # Generate an overview plot
                    jp52test.overview_plot(figfile=os.path.join(picdir,'test_model_b1_{0}_H_{1}_a1_{2}_b2_{3}.png'.format(beta1, Hval, alpha1val, beta2val)))
                    plt.show()

                    # Now construct a dataframe with some useful properties...
                    dfx = pull_out_group_vel_params(jp52test) # Function at the top of this script.
                    dfx['beta1']= beta1
                    dfx['H']=Hval
                    dfx['alpha1']= alpha1val
                    dfx['beta2']= beta2val
                    #print(dfx)
                    df = pd.concat([df,dfx],ignore_index=True)
                    del dfx

    print(df)
    df.to_csv('/Users/pn22327/Documents/foulnessdata/code/JP52_modelling/JP52_tempdf.csv')
    
