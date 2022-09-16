# Public API's
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning) # Turns off FutureWarnings
simplefilter(action='ignore', category=Warning)
import tensorflow as tf
import numpy as np
from scipy import stats
from helper_functions import *
from tqdm import tqdm

# Custom Imports
from Models.models import *
import config_file
from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from tqdm import tqdm
from keras.backend.tensorflow_backend import clear_session

model_name = 'encoder_repeats_1000'

#################################################### data load #########################################################
handler = data_handler(matlab_file = config_file.kamitani_data_mat, test_img_csv=config_file.test_image_ids, train_img_csv=config_file.train_image_ids)
# Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0, normalize=1)
Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0, normalize=1, scale='standard', remove_outliers=0)
labels_train, labels = handler.get_labels(imag_data = 0)

file= np.load(config_file.images_npz) #_56
X = file['train_images']
X_test_avg = file['test_images'] # images seem to be in same order as in image_test_id.csv

X= X[labels_train]
X_test = X_test_avg[labels]

NUM_VOXELS = Y.shape[1]

y_preds_raw = []
vc_raw = []
std_raw = []
path_to_files = '../outputs/g_encoder/5891692/trained_encoders/encoder_repeats_1000'
for i in tqdm(range(590, 1000)):
    clear_session()
    # Load Model
    enc_param = encoder_param(NUM_VOXELS)
    enc_param.drop_out = 0.5
    enc = encoder(enc_param)
    enc.load_weights(path_to_files + '/test_' + f"{i+1:04}" + '/model_weights.hdf5')
    y_pred_raw = enc.predict(X_test)
    np.savetxt( path_to_files + '/test_' + f"{i+1:04}" + '/y_pred_raw.csv', y_pred_raw)
    vc_temp = []
    for k in range(NUM_VOXELS):
        vc_temp.append(stats.pearsonr(y_pred_raw[:, k], Y_test[:, k])[0])
    np.savetxt(path_to_files + '/test_' + f"{i+1:04}" + '/vc_raw.csv', vc_temp)
    y_preds_raw.append(y_pred_raw)
    vc_raw.append(vc_temp)
    std = np.std(y_pred_raw, axis=0)
    np.savetxt(path_to_files + '/test_' + f"{i+1:04}" + '/std_raw.csv', std)
    std_raw.append(std)

y_preds_raw = np.array(y_preds_raw)
vc_raw = np.array(vc_raw)
std_raw = np.array(std_raw)
print(y_preds_raw.shape)
print(vc_raw.shape)
print(std_raw.shape)
