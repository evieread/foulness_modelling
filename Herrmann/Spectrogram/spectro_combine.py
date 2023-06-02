#!/usr/bin/env python

import os
import sys
import obspy as ob
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram, get_window

para = {'axes.labelsize': 14, 'font.size': 14, 'legend.fontsize': 12,
   'legend.title_fontsize':12,
 'xtick.labelsize': 14,'ytick.labelsize': 14, 'figure.subplot.left': 0.12,
 'figure.subplot.right': 0.98, 'figure.subplot.bottom': 0.11,
 'figure.subplot.top': 0.97}
plt.rcParams.update(para)


def ZRT_spectrograms_synth_real(sname,rname,filtlims,seg_s=[3,2.5],
                   xlims=None,vlims_s=None,vlims_r=None,
                   velticks=None,
                    shotname='XX',velunits=['syn',r'${\mu}$m/s'],
                   figfile=False):

    '''
    Expects:
    sname = one filename (string not list); synthetic filename
    rname = one filename (string not list)' data filename

    filtlims = a two-element list giving [min_filt,max_filt],
    [this filter is applied to both the synth and real data]

    seg_s = NOTE: This has changed from previous implementations
    This is a two element list giving: [window_length,overlap_length]
    window_length is the length of the spectrogram window in seconds
    overlap_length is the length of the overlap between windows in seconds
    - This has been changed to seconds, rather than number of samples, because
    the data and the synthetics have, in general, got different sampling rates.
    To ensure that the resolution of the synthetic and data spectrograms are the
    same, we input the window lengths in physical time units, and then convert to
    numbers of samples in the code.

    xlims = the time limits for the axes
    vlims_s = the z-axis limits for the synthetic spectrogram: [min,max]
    vlims_r = the z-axis limits for the data spectrogram: [min,max]

    velticks = a list of the velocities for which ticks are added above the top
    axis of the plot.

    shotname = the shot name, added to the data plot label

    velunits = the units wanted for the output plot. Assumed that the input data
    is in m/s, so the velunits for the data allows a factor to be applied.
    [Note: 16/02/23, currently we have not sorted out the synthetic data units]

    figfile: True or False. If True, saves a .png file. If False, plots to screen



    '''

    # Create Stream and Add the traces - synth first, then data
    st = ob.core.stream.Stream()

    st.append(ob.read(sname)[0])
    st[-1].stats.channel = os.path.basename(sname).split('.')[0].split('_')[-1]

    st.append(ob.read(rname)[0])
    st[-1].stats.channel = os.path.basename(rname).split('.')[0].split('_')[-1]

    # Set up the filters, and windows - should be same for both real and synthetic
    filtmin = filtlims[0]
    filtmax = filtlims[1]

    secondsperseg = seg_s[0]
    secondsolap = seg_s[1]

    noisefloor = 0.7

    # Units - assumes that velunits list is in order: synth, data

    velfac = [0,0] # Setting up as list, due to two values
    for i, velunit in enumerate(velunits):
        if velunit == 'm/s':
            velfac[i] = 1.
        elif velunit == 'syn':
            velfac[i] = 1e7
        elif velunit == 'mm/s':
            velfac[i] = 1000.
        elif velunit == r'${\mu}$m/s':
            velfac[i] = 1e6
        elif velunit == 'nm/s':
            velfac[i] = 1e9
        else:
            print('velunit not recognized')
            sys.exit()


    # Going to apply filter to stream - so that each trace is filtered identically

    st.detrend('linear')
    st.taper(max_percentage=0.05)
    st.filter('bandpass',freqmin=filtmin,freqmax=filtmax)


# Four axes (LW = low frequency waveform; HS = High Frequency Spectrogram)

    tr_wsynth = st[0]
    tr_wreal = st[1]

    fig = plt.figure(figsize=(8.5,8.5))
    ax_wsynth = fig.add_axes([0.16,0.12,0.65,0.14])
    ax_ssynth = fig.add_axes([0.16,0.26,0.65,0.24])
    ax_wreal = fig.add_axes([0.16,0.56,0.65,0.14])
    ax_sreal = fig.add_axes([0.16,0.70,0.65,0.24])
    ax_srealvel = ax_sreal.twiny()
    axcbar_sreal = fig.add_axes([0.84, 0.75, 0.02, 0.14])
    axcbar_ssynth = fig.add_axes([0.84, 0.31, 0.02, 0.14])

    ax_wsynth.plot(tr_wsynth.times()+tr_wsynth.stats.sac['b'],tr_wsynth.data*velfac[0],'k-',lw=0.8)
    ax_wreal.plot(tr_wreal.times()+tr_wreal.stats.sac['b'],tr_wreal.data*velfac[1],'k-',lw=0.8)

    nperseg = int(secondsperseg*tr_wreal.stats.sampling_rate)
    nolap = int(secondsolap*tr_wreal.stats.sampling_rate)
    f_sreal,t_sreal,Sxx_sreal = spectrogram(tr_wreal.data*velfac[1],fs=tr_wreal.stats.sampling_rate,
                   nperseg=nperseg,
                   noverlap=nolap,
                   nfft=2056,
                   detrend='constant',
                   window=get_window('hann',nperseg))

    nperseg = int(secondsperseg*tr_wsynth.stats.sampling_rate)
    nolap = int(secondsolap*tr_wsynth.stats.sampling_rate)
    f_ssynth,t_ssynth,Sxx_ssynth = spectrogram(tr_wsynth.data*velfac[0],fs=tr_wsynth.stats.sampling_rate,
                   nperseg=nperseg,
                   noverlap=nolap,
                   nfft=2056,
                   detrend='constant',
                   window=get_window('hann',nperseg))

    #PRINT - TO BE REMOVED
    # What this shows is that t_ssynth starts at t=1.5 (if seg_s=[3,2.5]) which makes sense
    # as the time associated with the middle of the first window is 1.5s (i.e., 3/2). This is
    # why the spectrogram is cut off at the lower times for the synthetic. We may be able to get
    # round this by starting the synthetic at a time < 0 ?
    # print('---------------')
    # print(np.size(t_ssynth))
    # print(np.min(t_ssynth))
    # print('---------------')

    # Finding min/max power levels LS - in region to be plotted!
    z_ssynth = np.log10(Sxx_ssynth)
    ii = np.where(f_ssynth<filtmax)[0][-1]
    print('spec shape',np.shape(z_ssynth),ii)
    minz_ssynth = np.amin(z_ssynth[:,:ii])
    maxz_ssynth = np.amax(z_ssynth[:,:ii])
    print('min/max z',minz_ssynth,maxz_ssynth)

    # Finding min/max power levels HS
    z_sreal = np.log10(Sxx_sreal)
    ii = np.where(f_sreal<filtmax)[0][-1]
    minz_sreal = np.amin(z_sreal[:,:ii])
    maxz_sreal = np.amax(z_sreal[:,:ii])
    print('min/max z',minz_sreal,maxz_sreal)

    if vlims_s is not None:
        im_ssynth = ax_ssynth.pcolormesh(t_ssynth+tr_wsynth.stats.sac['b'], f_ssynth, z_ssynth,
                                shading='flat',vmin=vlims_ssynth[0],
                                vmax=vlims_ssynth[1],cmap='viridis')
    else:
        im_ssynth = ax_ssynth.pcolormesh(t_ssynth+tr_wsynth.stats.sac['b'], f_ssynth, z_ssynth,
                                shading='flat',vmin=minz_ssynth+noisefloor*(maxz_ssynth-minz_ssynth),
                                vmax=maxz_ssynth,cmap='viridis')

    if vlims_r is not None:
        im_sreal = ax_sreal.pcolormesh(t_sreal+tr_wreal.stats.sac['b'], f_sreal, z_sreal,
                                shading='flat',vmin=vlims_sreal[0],
                                vmax=vlims_sreal[1],cmap='viridis')
    else:
        im_sreal = ax_sreal.pcolormesh(t_sreal+tr_wreal.stats.sac['b'], f_sreal, z_sreal,
                                shading='flat',vmin=minz_sreal+noisefloor*(maxz_sreal-minz_sreal),
                                vmax=maxz_sreal,cmap='viridis')


    fig.colorbar(im_sreal, cax=axcbar_sreal)
    fig.colorbar(im_ssynth, cax=axcbar_ssynth)

    ax_ssynth.set_ylim([0,filtmax])
    ax_sreal.set_ylim([0,filtmax])

    if xlims is not None:
        ax_ssynth.set_xlim([xlims[0],xlims[1]])
        ax_sreal.set_xlim([xlims[0],xlims[1]])
        ax_wsynth.set_xlim([xlims[0],xlims[1]])
        ax_wreal.set_xlim([xlims[0],xlims[1]])
        ax_srealvel.set_xlim([xlims[0],xlims[1]])

    ax_ssynth.set_ylabel('Freq. (Hz)')
    ax_sreal.set_ylabel('Freq. (Hz)')
    ax_wsynth.set_xlabel('Time (s)')
    ax_wsynth.set_ylabel('Velocity (cm/s)')
    ax_wreal.set_ylabel('Velocity ('+velunits[1]+')')

    if velticks is not None:
        veltimes = (tr_wreal.stats.sac['dist']*1000.)/np.asarray(velticks)
        print(veltimes)
        ax_srealvel.set_xticks(veltimes)
        ax_srealvel.set_xticklabels(velticks)
        ax_srealvel.set_xlabel('Velocity (m/s)',fontsize=12)
        ax_srealvel.tick_params(axis='x', which='major', labelsize=12)
    else:
        ax_srealvel.set_xticks([])

    ax_ssynth.set_xticklabels([])
    ax_sreal.set_xticklabels([])

    axcbar_ssynth.set_ylabel('log10 PSD (('+velunits[0]+r')$^2$/Hz)', rotation=90)
    axcbar_sreal.set_ylabel('log10 PSD (('+velunits[1]+r')$^2$/Hz)', rotation=90)

    labelx = -0.15
    axs = [ax_sreal,ax_ssynth,ax_wsynth,ax_wreal]
    for ax in axs:
        ax.yaxis.set_label_coords(labelx, 0.5)

    ax_wsynth.text(0.985,0.96,'{0:.1f} to {1:.1f}Hz'.format(filtmin,filtmax),ha='right',
                va='top',transform=ax_wsynth.transAxes,size=16)
    ax_wreal.text(0.985,0.96,'>{0:.1f}Hz'.format(filtmin),ha='right',
                va='top',transform=ax_wreal.transAxes,size=16)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax_sreal.text(0.985,0.96,'{0},{1},{2}'.format(shotname,tr_wreal.stats.station,tr_wreal.stats.channel),ha='right',
                va='top',transform=ax_sreal.transAxes,size=16,bbox=props)
    ax_sreal.text(0.015,0.96,'Data',ha='left',
                va='top',transform=ax_sreal.transAxes,size=16,bbox=props)
    ax_ssynth.text(0.015,0.96,'Model',ha='left',
                va='top',transform=ax_ssynth.transAxes,size=16,bbox=props)   
    ax_sreal.grid('both')
    ax_ssynth.grid('both')

    plt.show()

    # if figfile:
    #     fig.savefig('{0}_{1}_{2}_synth_spectro.png'.format(shotname,tr_wreal.stats.station,tr_wreal.stats.channel),bbox_inches='tight')
    #     #fig.savefig('Spectro_test'+'.png',bbox_inches='tight')
    # else:
    #     plt.show()
    # plt.close('all')


if __name__=='__main__':


    # Testing the code:
    # Directory in which seismograms can be found - here my synthetic seismogram is in
    # the same directory as the data seismogram. If not, you would have to change this
    # between the generation of rname and sname
    topdir = '/Users/pn22327/Documents/foulnessdata/code/Synthetic_code'
    #topdir = '/Users/pn22327/Documents/foulnessdata/100kg_Data/100kg_TR06-09_NOPB/'
    
    # Read in the data
    sacr = 'TR09_S2_CHZ.sac'
    rname = os.path.join(topdir,sacr)
    
    # Read in the synthetic
    #topdir = './'
    sacs = 'B00101Z00.sac'
    sname = os.path.join(topdir,sacs)

    # Set up some auxillary parameters
    xlims = [0,36]  
    velticks = [370,180,120]
    filtlims= [0.5,8]
    vlims_sreal = [0,4] 
    vlims_ssynth = [-4.5,-0.5]
    seg_in_seconds = [3,2.5]


    ZRT_spectrograms_synth_real(sname,rname,filtlims,seg_s=seg_in_seconds,
                   xlims=xlims,vlims_s=vlims_ssynth,vlims_r=vlims_sreal,
                   velticks=velticks,
                    shotname='S2',velunits=['syn','${\mu}$m/s'],
                   figfile=True)