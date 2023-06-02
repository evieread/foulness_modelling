#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def run_misfit(file_path):
    df = pd.read_csv(file_path)

    specific_value = 1000
    df['gvec1p25hz'] = np.nan_to_num(df['gvec1p25hz'], nan=specific_value)

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
        Z1[y_idx, x_idx] = (gvec1p25hz[i])
        Z1_r[y_idx, x_idx] = (gvec1p25hz_r - gvec1p25hz[i])

    A1 = np.zeros_like(X1)
    A1_r = np.zeros_like(X1)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        A1[y_idx, x_idx] = (gvecMinFreq[i])
        A1_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])

    B1 = np.zeros_like(X1)
    B1_r = np.zeros_like(X1)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        B1[y_idx, x_idx] = (gvecMin[i])
        B1_r[y_idx, x_idx] = (gvecMin_r - gvecMin[i])

    Z2 = np.zeros_like(X2)
    Z2_r = np.zeros_like(X2)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(beta2) == beta2[i])[0]
        Z2[y_idx, x_idx] = (gvec1p25hz[i])
        Z2_r[y_idx, x_idx] = (gvec1p25hz_r - gvec1p25hz[i])


    A2 = np.zeros_like(X2)
    A2_r = np.zeros_like(X2)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(beta2) == beta2[i])[0]
        A2[y_idx, x_idx] = (gvecMinFreq[i])
        A2_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])

    B2 = np.zeros_like(X2)
    B2_r = np.zeros_like(X2)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(beta2) == beta2[i])[0]
        B2[y_idx, x_idx] = (gvecMin[i])
        B2_r[y_idx, x_idx] = (gvecMin_r - gvecMin[i])

    Z3 = np.zeros_like(X3)
    Z3_r = np.zeros_like(X3)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        Z3[y_idx, x_idx] = (gvec1p25hz[i])
        Z3_r[y_idx, x_idx] = (gvec1p25hz_r - gvec1p25hz[i])

    A3 = np.zeros_like(X3)
    A3_r = np.zeros_like(X3)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        A3[y_idx, x_idx] = (gvecMinFreq[i])
        A3_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])

    B3 = np.zeros_like(X3)
    B3_r = np.zeros_like(X3)
    for i in range(len(H)):
        x_idx = np.where(np.unique(H) == H[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        B3[y_idx, x_idx] = (gvecMin[i])
        B3_r[y_idx, x_idx] = (gvecMin_r - gvecMin[i])

    Z4 = np.zeros_like(X4)
    Z4_r = np.zeros_like(X4)
    for i in range(len(alpha1)):
        x_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        y_idx = np.where(np.unique(beta2) == beta2[i])[0]
        Z4[y_idx, x_idx] = (gvec1p25hz[i])
        Z4_r[y_idx, x_idx] = (gvec1p25hz_r - gvec1p25hz[i])

    A4 = np.zeros_like(X4)
    A4_r = np.zeros_like(X4)
    for i in range(len(alpha1)):
        x_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        y_idx = np.where(np.unique(beta2) == beta2[i])[0]
        A4[y_idx, x_idx] = (gvecMinFreq[i])
        A4_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])

    B4 = np.zeros_like(X4)
    B4_r = np.zeros_like(X4)
    for i in range(len(alpha1)):
        x_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        y_idx = np.where(np.unique(beta2) == beta2[i])[0]
        B4[y_idx, x_idx] = (gvecMin[i])
        B4_r[y_idx, x_idx] = (gvecMin_r - gvecMin[i])   

    Z5 = np.zeros_like(X5)
    Z5_r = np.zeros_like(X5)
    for i in range(len(alpha1)):
        x_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        Z5[y_idx, x_idx] = (gvec1p25hz[i])
        Z5_r[y_idx, x_idx] = (gvec1p25hz_r - gvec1p25hz[i])


    A5 = np.zeros_like(X5)
    A5_r = np.zeros_like(X5)
    for i in range(len(alpha1)):
        x_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        A5[y_idx, x_idx] = (gvecMinFreq[i])
        A5_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])

    B5 = np.zeros_like(X5)
    B5_r = np.zeros_like(X5)
    for i in range(len(alpha1)):
        x_idx = np.where(np.unique(alpha1) == alpha1[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        B5[y_idx, x_idx] = (gvecMin[i])
        B5_r[y_idx, x_idx] = (gvecMin_r - gvecMin[i])

    Z6 = np.zeros_like(X6)
    Z6_r = np.zeros_like(X6)
    for i in range(len(beta2)):
        x_idx = np.where(np.unique(beta2) == beta2[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        Z6[y_idx, x_idx] = (gvec1p25hz[i])
        Z6_r[y_idx, x_idx] = (gvec1p25hz_r - gvec1p25hz[i])


    A6 = np.zeros_like(X6)
    A6_r = np.zeros_like(X6)
    for i in range(len(beta2)):
        x_idx = np.where(np.unique(beta2) == beta2[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        A6[y_idx, x_idx] = (gvecMinFreq[i])
        A6_r[y_idx, x_idx] = (gvecMinFreq_r - gvecMinFreq[i])

    B6 = np.zeros_like(X6)
    B6_r = np.zeros_like(X6)
    for i in range(len(beta2)):
        x_idx = np.where(np.unique(beta2) == beta2[i])[0]
        y_idx = np.where(np.unique(beta1) == beta1[i])[0]
        B6[y_idx, x_idx] = (gvecMin[i])
        B6_r[y_idx, x_idx] = (gvecMin_r - gvecMin[i])

    Misfit1 = ((1/gvecMinFreq_r)*np.abs(A1_r))+((1/gvecMin_r)*np.abs(B1_r))+((1/gvec1p25hz_r)*np.abs(Z1_r))
    Misfit2 = ((1/gvecMinFreq_r)*np.abs(A2_r))+((1/gvecMin_r)*np.abs(B2_r))+((1/gvec1p25hz_r)*np.abs(Z2_r))
    Misfit3 = ((1/gvecMinFreq_r)*np.abs(A3_r))+((1/gvecMin_r)*np.abs(B3_r))+((1/gvec1p25hz_r)*np.abs(Z3_r))
    Misfit4 = ((1/gvecMinFreq_r)*np.abs(A4_r))+((1/gvecMin_r)*np.abs(B4_r))+((1/gvec1p25hz_r)*np.abs(Z4_r))
    Misfit5 = ((1/gvecMinFreq_r)*np.abs(A5_r))+((1/gvecMin_r)*np.abs(B5_r))+((1/gvec1p25hz_r)*np.abs(Z5_r))
    Misfit6 = ((1/gvecMinFreq_r)*np.abs(A6_r))+((1/gvecMin_r)*np.abs(B6_r))+((1/gvec1p25hz_r)*np.abs(Z6_r))

    fig, axes = plt.subplots(3, 3)#, figsize=(9, 9))

    cax1 = axes[0, 0].pcolormesh(X1, Y1, Misfit1, cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[0, 0].set_xlabel('H')
    axes[0, 0].set_ylabel('alpha1')
    cbar1 = fig.colorbar(cax1, ax=axes[0, 0], shrink=0.5)
    cbar1.set_label('Misfit')

    cax2 = axes[1, 0].pcolormesh(X2, Y2, Misfit2,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[1, 0].set_xlabel('H')
    axes[1, 0].set_ylabel('beta2')
    cbar2 = fig.colorbar(cax2, ax=axes[1, 0], shrink=0.5)
    cbar2.set_label('Misfit')

    cax3 = axes[2, 0].pcolormesh(X3, Y3, Misfit3,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[2, 0].set_xlabel('H')
    axes[2, 0].set_ylabel('beta1')
    cbar3 = fig.colorbar(cax3, ax=axes[2, 0], shrink=0.5)
    cbar3.set_label('Misfit')

    cax4 = axes[1, 1].pcolormesh(X4, Y4, Misfit4,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[1, 1].set_xlabel('alpha1')
    axes[1, 1].set_ylabel('beta2')
    cbar4 = fig.colorbar(cax4, ax=axes[1, 1], shrink=0.5)
    cbar4.set_label('Misfit')

    cax5 = axes[2, 1].pcolormesh(X5, Y5, Misfit5,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[2, 1].set_xlabel('alpha1')
    axes[2, 1].set_ylabel('beta1')
    cbar5 = fig.colorbar(cax5, ax=axes[2, 1], shrink=0.5)
    cbar5.set_label('Misfit')

    cax6 = axes[2, 2].pcolormesh(X6, Y6, Misfit6,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axes[2, 2].set_xlabel('beta2')
    axes[2, 2].set_ylabel('beta1')
    cbar6 = fig.colorbar(cax6, ax=axes[2, 2], shrink=0.5)
    cbar6.set_label('Misfit')

    axes[0, 1].remove()
    axes[0, 2].remove()
    axes[1, 2].remove()

    fig.suptitle('Frequency at minimum group velocity + minimum group velocity + group velocity at 1.25Hz - residual', fontsize=16)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    run_misfit(file_path)