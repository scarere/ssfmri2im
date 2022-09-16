# Get rid of annoying warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning) # Turns off FutureWarnings
simplefilter(action='ignore', category=Warning)
import json
import yaml
from helper_functions import *
from scipy import stats
import time
import multiprocessing
import sys

# OG Imports
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session, clear_session, get_session
from keras.losses import mean_squared_error, cosine_proximity
from keras.optimizers import SGD
from Utils.batch_generator import *
from Utils.misc import download_network_weights
import sys
from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from Models.models import *
import config_file

if (os.path.exists(config_file.encoder_weights) and not config_file.retrain_encoder):
    print('pretrained encoder weights file exist')
    sys.exit()
else:
    print('training encoder')

# os.environ["CUDA_VISIBLE_DEVICES"] = config_file.GPU_ID
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu_config = tf.ConfigProto()
# gpu_config.gpu_options.allow_growth = True
# set_session(tf.Session(config=gpu_config))

# Set script params
experiment_name = 'encoder_repeats_1000'
training_params = {}
training_params['num_repeats'] = 1000
training_params['batch_size'] = 64
training_params['val_batch_size'] = 50
training_params['epochs'] = 80
training_params['roi'] = 'ROI_VC'
training_params['scale_type'] = 'standard'
training_params['outlier_removal'] = 0
training_params['normalization'] = 1

# loss = mean_squared_logarithmic_error
# training_params['loss'] = 'msle'

#################################################### data load ##########################################################
print('Script: encoder_train.py')
handler = data_handler(matlab_file = config_file.kamitani_data_mat, test_img_csv=config_file.test_image_ids, train_img_csv=config_file.train_image_ids)
handler.print_meta_desc()
Y,Y_test,Y_test_avg = handler.get_data(
    roi = training_params['roi'], 
    normalize=1,
    scale=training_params['scale_type'], 
    remove_outliers=training_params['outlier_removal']
    )

labels_train, labels = handler.get_labels()
NUM_VOXELS = Y.shape[1]

file= np.load(config_file.images_npz)
X = file['train_images']
X_test_avg = file['test_images']

X= X[labels_train]
x_train = X
X_test_all = X_test_avg[labels]
x_test = X_test_avg

y_train = Y
y_test = Y_test_avg
print('x_train: ', np.shape(x_train))
print('x_test: ', np.shape(x_test))
print('y_train: ', np.shape(y_train))
print('y_test: ', np.shape(y_test))

xyz, _ = handler.get_voxel_loc(roi=training_params['roi'])
print('XYZ: ', np.shape(xyz))

#################################################### losses & schedule ##########################################################

#################################################### model ##########################################################
if not (~os.path.exists(config_file.caffenet_models_weights)):
    download_network_weights()
else:
    print("pre-trained matconvnet model is already downloaded")

for i in range(training_params['num_repeats']):
  clear_session() # Clear previous session memory to prevent OOM
  training_params['lr_schedule'] = {
    'initial_lr':0.1,
    'gamma':0.1,
    'milestones': [20, 35, 45, 50]}

  def step_decay(epoch):
      lrate =training_params['lr_schedule']['initial_lr']
      for milestone in training_params['lr_schedule']['milestones']:
          if epoch >= milestone:
              lrate = training_params['lr_schedule']['gamma']*lrate
      return lrate

  def combined_loss(y_true, y_pred):
      return mean_squared_error(y_true, y_pred) +  0.1*cosine_proximity(y_true, y_pred)

  loss = combined_loss
  training_params['loss'] = 'combined_loss'
  print('Starting Test ', i+1, flush=True)
  ts = time.time()
  # Create Model
  model_name = 'test_' + f"{i+1:04}"
  enc_param = encoder_param(NUM_VOXELS)
  enc_param.drop_out = 0.5
  vision_model = encoder(enc_param)

  optimizer = SGD(lr=training_params['lr_schedule']['initial_lr'], decay = 0.0 , momentum = 0.9,nesterov=True)
  vision_model.compile(loss=loss, optimizer=optimizer,metrics=['mse','cosine_proximity','mae', combined_loss])

  # Callbacks
  if not os.path.exists('trained_encoders/' + model_name):
      os.makedirs(os.path.join('trained_encoders', experiment_name, model_name))

  model_checkpoint_callback = ModelCheckpoint(
      filepath=os.path.join('trained_encoders', experiment_name, model_name, 'best_ckpt.hdf5'),
      save_weights_only=True,
      monitor='val_loss',
      mode='min',
      save_best_only=True
      )
  callbacks = [LearningRateScheduler(step_decay, verbose=0)]
  callbacks.append(model_checkpoint_callback)

  # Training
  train_generator = batch_generator_enc(x_train, y_train, batch_size=training_params['batch_size'],max_shift = 5)
  test_generator = batch_generator_enc(x_test, y_test, batch_size=training_params['val_batch_size'],max_shift = 0)
  training = vision_model.fit_generator(train_generator, epochs=training_params['epochs'],validation_data=test_generator ,verbose=0,use_multiprocessing=False,callbacks=callbacks) #, steps_per_epoch=1200//64 , validation_steps=1

  # Evaluation
  y_pred = vision_model.predict(x_test)
  std = np.std(y_pred, axis=0)
  vc = []
  for v in range(NUM_VOXELS):
    vc.append(stats.pearsonr(y_pred[:, v], y_test[:, v])[0])
  vc = np.array(vc)


  # Save
  vision_model.save_weights(os.path.join('trained_encoders', experiment_name, model_name, 'model_weights.hdf5'))
  np.savetxt(os.path.join('trained_encoders', experiment_name, model_name, 'y_pred.csv'), y_pred, delimiter=',')
  np.savetxt(os.path.join('trained_encoders', experiment_name, model_name, 'voxel_corr.csv'), vc, delimiter=',')
  np.savetxt(os.path.join('trained_encoders', experiment_name, model_name, 'voxel_std.csv'), std, delimiter=',')

  if 'lr' in training.history.keys():
        training.history['lr'] = [np.float64(lr) for lr in training.history['lr']]

  with open(os.path.join('trained_encoders', experiment_name, model_name, 'train_history.json'), 'w') as file:
    json.dump(training.history, file) # save training history
  te = time.time() - ts
  print('Finished test', i+1, ' after ', int(te), 's', flush=True)

  # delete vars
  del vision_model
  del model_checkpoint_callback
  del callbacks
  del train_generator
  del test_generator
  del training
  del enc_param
  del optimizer

# Save training params
optimizer = SGD(lr=training_params['lr_schedule']['initial_lr'], decay = 0.0 , momentum = 0.9,nesterov=True)
training_params['optimizer'] = optimizer.get_config()
with open('trained_encoders/' + experiment_name + '/train_config.yaml', 'w') as file:
  yaml.dump(training_params, file)
