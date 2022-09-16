# Public API's
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning) # Turns off FutureWarnings
simplefilter(action='ignore', category=Warning)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# Custom Imports
from Models.models import *
import config_file
from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler

template = 'KamitaniData/func_raw/sub-03_ses-perceptionTest01_task-perception_run-01_bold_preproc.nii.gz'
model_name = 'encoder_epochs_5'

#################################################### data load #########################################################
handler = data_handler(matlab_file = config_file.kamitani_data_mat)
# Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0, normalize=1)
Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0, normalize=1, scale='standard', remove_outliers=0)
labels_train, labels = handler.get_labels(imag_data = 0)

print(labels)
file= np.load(config_file.images_npz) #_56
X = file['train_images']
X_test_avg = file['test_images'] # images seem to be in same order as in image_test_id.csv

X= X[labels_train]
X_test = X_test_avg[labels]

NUM_VOXELS = Y.shape[1]

# Get voxel coordinates
xyz, _ = handler.get_voxel_loc(roi='ROI_VC')

# Load Model
enc_param = encoder_param(NUM_VOXELS)
enc_param.drop_out = 0.5
enc = encoder(enc_param)
print(enc.summary())

y_test = Y_test_avg

for i in range(90):
  enc.load_weights('trained_encoders/' + model_name + '/ckpt_step_' + f"{i+1:02}" + '.hdf5')
  y_pred = enc.predict(X_test_avg)
  voxel_corr = []
  for j in range(NUM_VOXELS):
      voxel_corr.append(stats.pearsonr(y_pred[:, j], y_test[:, j])[0])
  stds = np.std(y_pred, axis=0)
  vc_std = [voxel_corr, stds]
  np.savetxt('trained_encoders/' + model_name + '/vc_std_step_' + f"{i+1:02}" + '.csv', vc_std, delimiter=',')

fig, ax = plt.subplots()

def init():
  ax.set_xlim(0, 0.7)
  ax.set_ylim(-0.6, 1)

def update(frame):
  i = frame + 1
  data = np.loadtxt('trained_encoders/' + model_name + '/vc_std_step_' + f"{i+1:02}" + '.csv', delimiter=',')
  vc = data[0]
  std = data[1]
  plt.clf()
  #plt.hexbin(std, vc, cmap='Blues', bins='log', gridsize=50)
  plt.scatter(std, vc, s=0.5, cmap='gist_ncar', c=np.linspace(0, 1, NUM_VOXELS))
  plt.text(0.5, 0, 'Step: ' + str(i))
  plt.ylim(-0.6, 1)
  plt.xlim(0, 0.7)


ani = FuncAnimation(fig, update, frames=90, init_func=init)
# plt.show()

save_path = 'trained_encoders/' + model_name + '/vc_std_scatter.gif'
mpeg_writer = FFMpegWriter(fps=2)
gif_writer = PillowWriter(fps=3)
ani.save(save_path, writer=gif_writer)