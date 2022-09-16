# Public API's
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning) # Turns off FutureWarnings
simplefilter(action='ignore', category=Warning)
import tensorflow as tf
import keras
import numpy as np
from bdpy.mri.image import export_brain_image
import nibabel
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress
from helper_functions import filter_by_max, remove_low_corr_voxels_v1, remove_low_corr_voxels_v2, remove_low_variance_voxels
import yaml
import json

# Custom Imports
from Models.models import *
import config_file
from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler

template = 'KamitaniData/func_raw/sub-03_ses-perceptionTest01_task-perception_run-01_bold_preproc.nii.gz'
model_name = 'encoder_q_n5'

#################################################### data load #########################################################
handler = data_handler(matlab_file = config_file.kamitani_data_mat)
Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0, normalize=1)
labels_train, labels = handler.get_labels(imag_data = 0)

file= np.load(config_file.images_npz) #_56
X = file['train_images']
X_test_avg = file['test_images']

X= X[labels_train]
X_test = X_test_avg[labels]

NUM_VOXELS = Y.shape[1]

# Get voxel coordinates
xyz, _ = handler.get_voxel_loc()

# Get hc voxels only
y_pred_path = 'trained_encoders/encoder/y_pred.csv'
y_test = Y_test_avg

with open("trained_encoders/" + model_name + '/train_config.yaml', 'r') as file:
    training_params = yaml.safe_load(file)

print(training_params['bins'])

# Quantize fmri data
y_test = np.digitize(y_test, training_params['bins'])
y_test = y_test - int(len(training_params['bins']) / 2)
y_test = y_test.astype('float32')
vals, counts = np.unique(y_test, return_counts=True)

# y_train, y_test, xyz = remove_low_corr_voxels_v1(Y, Y_test_avg, xyz, threshold=0.4, pred_path=y_pred_path)

# y_train, y_val, y_test, xyz = remove_low_corr_voxels_v2(Y, Y_test_avg, xyz, threshold=0.4, pred_path=y_pred_path)
# num_samples = np.shape(Y_test_avg)[0]
# x_train = X
# x_val = X_test_avg[:int(num_samples*0.8), :, :]
# x_test_final = X_test_avg[int(num_samples*0.8):, :, :]

# NUM_VOXELS = y_train.shape[1]
# y_train, y_test, xyz = remove_low_variance_voxels(Y, Y_test_avg, xyz, threshold=1.0)
# NUM_VOXELS = y_train.shape[1]


# Load Model
enc_param = encoder_param(NUM_VOXELS)
enc_param.drop_out = 0.25
enc = encoder(enc_param)
enc.load_weights('trained_encoders/' + model_name + '/model_weights.hdf5')

# Get fmri reconstruction
# sample = np.expand_dims(X_test[0], axis=0)
# recon = enc.predict(sample)s
y_pred = enc.predict(X_test_avg)
np.savetxt('trained_encoders/' + model_name + '/y_pred.csv', y_pred, delimiter=',')
#y_pred = np.loadtxt('trained_encoders/' + model_name + '/y_pred.csv', delimiter=',')
# y_pred = np.loadtxt('trained_encoders/' + model_name + '/y_pred.csv', delimiter=',')


# test_mean = np.mean(y_test, axis=0)
# train_mean = np.mean(Y, axis=0)
# print(np.shape(train_mean))

# nifti = export_brain_image(y_test[0], template=template, xyz=xyz)
# nibabel.save(nifti, 'nifti/' + model_name + '/sub03_s0_groundtruth_qn9.nii.gz')
# nibabel.save(nifti, 'nifti//sub03_test_mean.nii.gz')


