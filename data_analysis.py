# Public API's
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning) # Turns off FutureWarnings
simplefilter(action='ignore', category=Warning)
import tensorflow as tf
import numpy as np
from bdpy.mri.image import export_brain_image
import nibabel
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
from helper_functions import *
import yaml
from nilearn import plotting
from keras.losses import cosine_proximity
import matplotlib.cm as cm
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

# Custom Imports
from Models.models import *
import config_file
from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler


template = 'KamitaniData/func_raw/sub-03_ses-perceptionTest01_task-perception_run-01_bold_preproc.nii.gz'
model_name = 'encoder'

#################################################### data load #########################################################
handler = data_handler(matlab_file = config_file.kamitani_data_mat)
# Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0, normalize=1)
Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0, normalize=0, scale='standard', remove_outliers=0)
labels_train, labels = handler.get_labels(imag_data = 0)

file= np.load(config_file.images_npz) #_56
X = file['train_images']
X_test_avg = file['test_images'] # images seem to be in same order as in image_test_id.csv

X= X[labels_train]
X_test = X_test_avg[labels]

NUM_VOXELS = Y.shape[1]
# bvc = np.loadtxt('beliy_vox_corr.csv', delimiter=',')
# idx = np.where(bvc > 0)[0]
# y_train = Y[:, idx]
# y_test = Y_test_avg[:, idx]
# NUM_VOXELS = y_train.shape[1]
y_test = Y_test_avg
y_train = Y

# Get voxel coordinates
xyz, _ = handler.get_voxel_loc(roi='ROI_VC')

# data = loadmat('trained_encoders/encoder_noisy_test/split_data.mat')
# x_train = data['x_train']
# x_test = data['x_test']
# y_train = data['y_train']
# y_test = data['y_test']

# Load Model
enc_param = encoder_param(NUM_VOXELS)
enc_param.drop_out = 0.5
#enc_param.conv_ch = 192
enc = encoder(enc_param)
#print(enc.summary())
enc.load_weights('trained_encoders/' + model_name + '/model_weights.hdf5')
#enc.load_weights('trained_encoders/' + model_name + '/best_ckpt.hdf5')

# y_pred = enc.predict(X_test_avg)
#y_pred_train = enc.predict(X)
# np.savetxt('trained_encoders/' + model_name + '/y_pred.csv', y_pred, delimiter=',')
y_pred = np.loadtxt('trained_encoders/' + model_name + '/y_pred.csv', delimiter=',')
# y_pred = np.broadcast_to(np.mean(y_train, axis=0), shape=y_test.shape)
# y_pred = np.add(y_pred, np.random.normal(0, 0.05, y_pred.shape))

# w, b = enc.layers[-1].get_weights()
# size = list_prod(np.shape(w)[:-1])
# w = np.reshape(w, [size, NUM_VOXELS])
# print(np.shape(w))
# w = w[:, idx]
# w_flat = w.flatten()
# print(np.std(w_flat))
# plt.hist(w_flat, bins=100)
# plt.show()

correlation = []
voxel_corr = []
vc_train = []
vc_p = []

for i in tqdm(range(NUM_VOXELS)):
    voxel_corr.append(stats.pearsonr(y_pred[:, i], y_test[:, i])[0])
    #vc_train.append(stats.pearsonr(y_pred_train[:, i], y_train[:, i])[0])
    vc_p.append(stats.pearsonr(y_pred[:, i], y_test[:, i])[1])
for i in tqdm(range(50)):
    correlation.append(stats.pearsonr(y_pred[i], y_test[i])[0]) # returns correlation coefficient r and two tailed p-value

voxel_corr = np.array(voxel_corr)
vc_p = np.array(vc_p)
print('Sample Corr: ', np.mean(correlation))
print('Voxel Corr: ', np.mean(voxel_corr))

stds = np.std(y_pred, axis=0)
stdtest = np.std(y_test, axis=0)
print(np.min(voxel_corr))

# vp = np.where(voxel_corr > 0)
# vn = np.where(voxel_corr < 0)

# vc_std = voxel_corr * stds
# vc_std_norm = voxel_corr * stds/stdtest
# print(np.min(vc_std))

# vt = np.where(vc_std > abs(np.min(vc_std)))
# vb = np.where(vc_std < abs(np.min(vc_std)))
# print(len(vt[0]))
# print(np.mean(vc_std[vt]))
# print(np.mean(vc_std_norm[vt]))
# print(np.mean(voxel_corr[vt]))
# print(np.mean(stds[vt]))
# print(len(np.where(vc_p < 0.0001)[0]))
# plt.scatter(stds[vt], voxel_corr[vt], s=0.5)
# plt.scatter(stds[vb], voxel_corr[vb], s=0.5)
# plt.show()


import matplotlib
matplotlib.rcParams.update({'font.size': 16})

# b = np.arange(-0.5, 1, 0.02)

# plt.hist(voxel_corr[np.where( (vc_p < 0.05) & (vc_p > 0.001))], bins=b, color='cornflowerblue',)
# plt.hist(voxel_corr[np.where(vc_p > 0.05)], bins=b, color='lightsteelblue',)
# plt.hist(voxel_corr[np.where(vc_p < 0.001)], bins=b, color='royalblue',)

# plt.hist((voxel_corr[np.where(vc_p < 0.001)], voxel_corr[np.where(vc_p > 0.05)], voxel_corr[np.where( (vc_p < 0.05) & (vc_p > 0.001))]), bins=b, stacked=True, color=['royalblue', 'lightsteelblue', 'cornflowerblue'])
# plt.xlabel('Voxelwise Correlation')
# plt.ylabel('Number of Voxels')
# plt.legend(['p > 0,05', '0.05 > p > 0.0001', 'p < 0.0001'])
# plt.axvline(0, color='black', alpha=0.5, linewidth=1, linestyle='--')
# plt.show()

#Critical Values for 50 samples
# p=0.001 @ r=0.4514
# p=0.0001 r=0.5223

# for 20 samples
# p=0.001 @ r=0.6788

# stds = np.loadtxt('beliy_vox_stds.csv', delimiter=',')
# plt.scatter(voxel_corr, bvc[idx], c=stds[idx], s=2, cmap='viridis', vmax=0.2)
# plt.colorbar()
# plt.plot([-0.5, 1], [-0.5, 1], '--', color='red')
# plt.axvline(0, alpha=0.5, color='black', linewidth=1)
# plt.axhline(0, alpha=0.5, color='black', linewidth=1)
# plt.xlabel('Trained on Positive Voxel Subset')
# plt.ylabel('Trained with All Voxels')
# plt.xlim(-0.5, 0.6)
# plt.ylim(0, 0.6)
# plt.tight_layout()
# plt.show()
# plt.hist(voxel_corr, bins=50)
# plt.xlabel('Voxelwise Correlation', fontsize=16)
# plt.ylabel('Number of Voxels', fontsize=16)
# plt.xlim(-0.5, 1)
# vc_p = np.array(vc_p)
# plt.xlabel('P-Value', fontsize=16)
# plt.ylabel('Voxelwise Correlation', fontsize=16)
# plt.xlim(0, 0.05)
# plt.hist(vc_p[voxel_corr>0], bins=50)
# plt.tight_layout()
# plt.show()
#plt.savefig('~/downloads/beliy_vox_corr.png')

# pvals = np.loadtxt('beliy_pvals.csv', delimiter=',')
# pvals = pvals[idx]
# stdp = stds[idx]
# bvcp = bvc[idx]

# vc2 = voxel_corr[np.where( (pvals < 0.05) & (pvals > 0.0001))[0]]
# bvcp2 = bvcp[np.where( (pvals < 0.05) & (pvals > 0.0001))]
# stdp2 = stdp[np.where( (pvals < 0.05) & (pvals > 0.0001))]
# plt.scatter(vc2 - bvcp2, stdp2, s=5, c=bvcp2, cmap='magma')
# plt.scatter(voxel_corr - bvc[idx], stds[idx], s=2, c=bvc[idx], cmap='magma')
# plt.xlabel('Difference in Voxelwise Correlation (r2 - r1)')
# plt.ylabel('Voxelwise standard deviation')

# plt.scatter(voxel_corr - bvc[idx], np.minimum(bvc[idx], voxel_corr), s=2, c=stds[idx], cmap='Set1', vmax=0.2, vmin=0)
# cbar = plt.colorbar()
# cbar.ax.set_ylabel('Original Voxelwise correlation')
# plt.show()


# plt.subplot(1, 3, 1)
# plt.hexbin(stds, v_ken, gridsize=50, bins='log', vmax=50, cmap='Blues')
# plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.7)
# # plt.ylim(-0.6, 1)
# # plt.xlim(0, 0.6)
# plt.subplot(1, 3, 2)
# plt.ylim(-0.6, 1)
# plt.xlim(0, 0.6)
# plt.hexbin(stds, voxel_corr, gridsize=60, bins='log', vmax=50, cmap='Blues', extent=(0, 0.6, -0.6, 1))
#plt.scatter(stds, voxel_corr, s=1)
# plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.7)
# plt.subplot(1, 3, 3)
# plt.hexbin(stds, v_spear, gridsize=50, bins='log', vmax=50, cmap='Blues')
# plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.7)
# plt.ylim(-0.6, 1)
# plt.xlim(0, 0.6)
# plt.xlabel('Voxelwise Standard Deviation')
# plt.ylabel('Voxelwise Correlation')
# plt.tight_layout()
# plt.show()

# plt.scatter(voxel_corr, vox_mi, s=0.5, c=stds, cmap='hsv')
# plt.colorbar()
# plt.subplot(1, 2, 1)
# plt.scatter(stds, voxel_corr, s=2, c=bvc[idx], cmap='magma')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.scatter(stds, vox_mi, s=0.5)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(stds, vox_mi, voxel_corr, marker='o')
# plt.show()

# p = voxel_corr > 0
# n = voxel_corr > 0
# vc_p = np.array(vc_p)
# print((vc_p[p] > 0.001).sum())

# d = np.stack([stds, voxel_corr], axis=-1)
# km = GaussianMixture(n_components=2, max_iter=10000, covariance_type='diag').fit(d)
# labels = km.predict(d)
# i = np.where(labels==1)
# plt.scatter(stds[i], voxel_corr[i], s=0.5)
# i = np.where(labels==0)
# plt.scatter(stds[i], voxel_corr[i], s=0.5)
# plt.show()