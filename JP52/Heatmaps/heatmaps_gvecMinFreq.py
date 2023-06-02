#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def run_gvecminfreq(file_path):
    df = pd.read_csv(file_path)

    H = df.iloc[:, 1]
    alpha1 = df.iloc[:, 2]
    beta1 = df.iloc[:, 3]
    beta2 = df.iloc[:, 4]
    branch3Freq = df.iloc[:, 5]
    gvec1hz = df.iloc[:, 6]
    gvec1p25hz = df.iloc[:, 7]
    gvecMin = df.iloc[:, 8]
    gvecMinFreq = df.iloc[:, 9]

    gvecMin_r = 120
    gvecMinFreq_r = 1.65
    branch3Freq_r = 1.8
    gvec1hz_r = 400
    gvec1p25hz_r = 275

    X1, Y1 = np.meshgrid(np.unique(H), np.unique(alpha1))
    X2, Y2 = np.meshgrid(np.unique(H), np.unique(beta2))
    X3, Y3 = np.meshgrid(np.unique(H), np.unique(beta1))
    X4, Y4 = np.meshgrid(np.unique(alpha1), np.unique(beta2))
    X5, Y5 = np.meshgrid(np.unique(alpha1), np.unique(beta1))
    X6, Y6 = np.meshgrid(np.unique(beta2), np.unique(beta1))

    Z1 = np.zeros_like(X1)
    Z1_r = np.zeros_like(X1)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        Z1[y_idx, x_idx] = (gvecMinFreq[i])
        Z1_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])


    Z2 = np.zeros_like(X2)
    Z2_r = np.zeros_like(X2)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(beta2) == beta2[i])[0]
        Z2[y_idx, x_idx] = (gvecMinFreq[i])
        Z2_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])


    Z3 = np.zeros_like(X3)
    Z3_r = np.zeros_like(X3)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        Z3[y_idx, x_idx] = (gvecMinFreq[i])
        Z3_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])

    Z4 = np.zeros_like(X4)
    Z4_r = np.zeros_like(X4)
    for i in range(len(alpha1)):
        x_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        y_idx = np.where(np.unique(beta2) == beta2[i])[0]
        Z4[y_idx, x_idx] = (gvecMinFreq[i])
        Z4_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])

    Z5 = np.zeros_like(X5)
    Z5_r = np.zeros_like(X5)
    for i in range(len(alpha1)):
        x_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        Z5[y_idx, x_idx] = (gvecMinFreq[i])
        Z5_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])

    Z6 = np.zeros_like(X6)
    Z6_r = np.zeros_like(X6)
    for i in range(len(beta2)):
        x_idx = np.where(np.unique(beta2) == beta2[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        Z6[y_idx, x_idx] = (gvecMinFreq[i])
        Z6_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])


    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    cax1 = axes[0, 0].pcolormesh(X1, Y1, Z1_r, cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[0, 0].set_xlabel('H')
    axes[0, 0].set_ylabel('alpha1')
    cbar1 = fig.colorbar(cax1, ax=axes[0, 0], shrink=0.5)
    cbar1.set_label('gvecMinFreq_residual (Hz)')

    cax2 = axes[1, 0].pcolormesh(X2, Y2, Z2_r,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[1, 0].set_xlabel('H')
    axes[1, 0].set_ylabel('beta2')
    cbar2 = fig.colorbar(cax2, ax=axes[1, 0], shrink=0.5)
    cbar2.set_label('gvecMinFreq_residual (Hz)')

    cax3 = axes[2, 0].pcolormesh(X3, Y3, Z3_r,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[2, 0].set_xlabel('H')
    axes[2, 0].set_ylabel('beta1')
    cbar3 = fig.colorbar(cax3, ax=axes[2, 0], shrink=0.5)
    cbar3.set_label('gvecMinFreq_residual (Hz)')

    cax4 = axes[1, 1].pcolormesh(X4, Y4, Z4_r,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[1, 1].set_xlabel('alpha1')
    axes[1, 1].set_ylabel('beta2')
    cbar4 = fig.colorbar(cax4, ax=axes[1, 1], shrink=0.5)
    cbar4.set_label('gvecMinFreq_residual (Hz)')

    cax5 = axes[2, 1].pcolormesh(X5, Y5, Z5_r,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[2, 1].set_xlabel('alpha1')
    axes[2, 1].set_ylabel('beta1')
    cbar5 = fig.colorbar(cax5, ax=axes[2, 1], shrink=0.5)
    cbar5.set_label('gvecMinFreq_residual (Hz)')

    cax6 = axes[2, 2].pcolormesh(X6, Y6, Z6_r,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[2, 2].set_xlabel('beta2')
    axes[2, 2].set_ylabel('beta1')
    cbar6 = fig.colorbar(cax6, ax=axes[2, 2], shrink=0.5)
    cbar6.set_label('gvecMinFreq_residual (Hz)')

    axes[0, 1].remove()
    axes[0, 2].remove()
    axes[1, 2].remove()

    fig.suptitle('Frequency at minimum group velocity residual', fontsize=16)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    run_gvecminfreq(file_path)