#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('tempdf.csv')

H = df.iloc[:, 1]
beta1 = df.iloc[:, 2]
branch3Freq = df.iloc[:, 3]
gvec1hz = df.iloc[:, 4]
gvec1p25hz = df.iloc[:, 5]
gvecMin = df.iloc[:, 6]
gvecMinFreq = df.iloc[:, 7]


gvecMin_r = 120
gvecMinFreq_r = 1.65
branch3Freq_r = 1.8
gvec1hz_r = 400
gvec1p25hz_r = 275

X, Y = np.meshgrid(np.unique(H), np.unique(beta1))


Z1 = np.zeros_like(X)
Z1_r = np.zeros_like(X)
for i in range(len(H)):
    x_idx = np.where(np.unique(H) == H[i])[0]
    y_idx = np.where(np.unique(beta1) == beta1[i])[0]
    Z1[y_idx, x_idx] = (branch3Freq[i])
    Z1_r[y_idx, x_idx] = (branch3Freq_r - branch3Freq[i])


Z2 = np.zeros_like(X)
Z2_r = np.zeros_like(X)
for i in range(len(H)):
    x_idx = np.where(np.unique(H) == H[i])[0]
    y_idx = np.where(np.unique(beta1) == beta1[i])[0]
    Z2[y_idx, x_idx] = (gvecMinFreq[i])
    Z2_r[y_idx, x_idx] = (gvecMinFreq[i] - gvecMinFreq_r)


Z3 = np.zeros_like(X)
Z3_r = np.zeros_like(X)
for i in range(len(H)):
    x_idx = np.where(np.unique(H) == H[i])[0]
    y_idx = np.where(np.unique(beta1) == beta1[i])[0]
    Z3[y_idx, x_idx] = (gvecMin[i])
    Z3_r[y_idx, x_idx] = (gvecMin[i] - gvecMin_r)

Z4 = np.zeros_like(X)
Z4_r = np.zeros_like(X)
for i in range(len(H)):
    x_idx = np.where(np.unique(H) == H[i])[0]
    y_idx = np.where(np.unique(beta1) == beta1[i])[0]
    Z4[y_idx, x_idx] = (gvec1hz[i])
    Z4_r[y_idx, x_idx] = (gvec1hz[i] - gvec1hz_r)

Z5 = np.zeros_like(X)
Z5_r = np.zeros_like(X)
for i in range(len(H)):
    x_idx = np.where(np.unique(H) == H[i])[0]
    y_idx = np.where(np.unique(beta1) == beta1[i])[0]
    Z5[y_idx, x_idx] = (gvec1p25hz[i])
    Z5_r[y_idx, x_idx] = (gvec1p25hz[i] - gvec1p25hz_r)

Misfit_gvecMin= ((1/gvecMinFreq_r)*np.abs(Z2_r))+((1/gvecMin_r)*np.abs(Z3_r))

Misfit_gvec1hz = ((1/gvecMinFreq_r)*np.abs(Z2_r))+((1/gvecMin_r)*np.abs(Z3_r))+((1/gvec1p25hz_r)*np.abs(Z5_r))
#((1/gvec1hz_r)*np.abs(Z4_r))

num_x = len(np.unique(H))
num_y = len(np.unique(beta1))
aspect_ratio = num_y / num_x

fig, axes = plt.subplots(1, 5, figsize=(22, 4))

cax1 = axes[0].pcolormesh(X, Y, Z1_r, cmap='seismic', vmin=-1, vmax=1, shading='auto')
axes[0].set_xlabel('H')
axes[0].set_ylabel('beta1')
axes[0].set_aspect(aspect_ratio)
cbar1 = fig.colorbar(cax1, ax=axes[0], shrink=0.5)
cbar1.set_label('branch3Freq_residual (Hz)')

cax2 = axes[1].pcolormesh(X, Y, Z2_r,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
axes[1].set_xlabel('H')
axes[1].set_ylabel('beta1')
axes[1].set_aspect(aspect_ratio)
cbar2 = fig.colorbar(cax2, ax=axes[1], shrink=0.5)
cbar2.set_label('gvecMinFreq_residual (Hz)')

cax3 = axes[2].pcolormesh(X, Y, Z3_r,  cmap='seismic', vmin=-50, vmax=50, shading='auto')
axes[2].set_xlabel('H')
axes[2].set_ylabel('beta1')
axes[2].set_aspect(aspect_ratio)
cbar3 = fig.colorbar(cax3, ax=axes[2], shrink=0.5)
cbar3.set_label('gvecMin_residual (Hz)')

cax4 = axes[3].pcolormesh(X, Y, Misfit_gvecMin,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
axes[3].set_xlabel('H')
axes[3].set_ylabel('beta1')
axes[3].set_aspect(aspect_ratio)
cbar4 = fig.colorbar(cax4, ax=axes[3], shrink=0.5)
cbar4.set_label('Misfit_gvecMin')

cax5 = axes[4].pcolormesh(X, Y, Misfit_gvec1hz,  cmap='seismic', vmin=-1, vmax=1, shading='auto')
axes[4].set_xlabel('H')
axes[4].set_ylabel('beta1')
axes[4].set_aspect(aspect_ratio)
cbar5 = fig.colorbar(cax5, ax=axes[4], shrink=0.5)
cbar5.set_label('Misfit_gvecfull')


plt.tight_layout()
plt.show()