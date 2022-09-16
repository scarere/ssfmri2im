# Get rid of annoying warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning) # Turns off FutureWarnings
simplefilter(action='ignore', category=Warning)
import json
import yaml
import pandas as pd
from helper_functions import *
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import config_file as config

# OG Imports
import copy
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.losses import mean_squared_error, cosine_proximity, mean_squared_logarithmic_error
from keras.optimizers import SGD, Optimizer, Adam
from Utils.batch_generator import *
from Utils.misc import download_network_weights
import  tensorflow as tf
from Models.models import *
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = config_file.GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
set_session(tf.Session(config=gpu_config))

# Set script params
save = True
model_name = 'digit_encoder_test8'
training_params = {}
training_params['batch_size'] = 8
training_params['val_batch_size'] = 20
training_params['epochs'] = 1200
training_params['roi'] = 'ROI_VC'
training_params['scale_type'] = 'standard'
# training_params['outlier_removal'] = 1
# training_params['normalization'] = 1

# loss = mean_squared_logarithmic_error
# training_params['loss'] = 'msle'

# initial_lrate = 0.1
# optimizer = SGD(lr=initial_lrate, decay = 0.0 , momentum = 0.9,nesterov=True)
initial_lrate = 0.001
optimizer = Adam(lr=initial_lrate)
training_params['optimizer'] = optimizer.get_config()

#################################################### data load ##########################################################
print('Script: encoder_train.py')
print('Loading Data...', flush=True)
dataset = loadmat(os.path.join(config.project_dir, 'handwritten_digits_in_fmri_dataset/69dataset_split.mat'))
x_test = dataset['x_test']
y_test = dataset['y_test']
x_train = dataset['x_train']
y_train = dataset['y_train']
x = dataset['x_all']
y = dataset['y_all']

#scale data
scaler = StandardScaler()
y_train = scaler.fit_transform(y_train)
y_test  = scaler.transform(y_test)
y  = scaler.fit_transform(y)

NUM_VOXELS=y_train.shape[1]
print(x_train.dtype)
print('Model Name: ', model_name)

#################################################### losses & schedule ##########################################################

training_params['lr_schedule'] = True
if training_params['lr_schedule']:
    training_params['lr_schedule'] = {
        'gamma':0.1,
        'milestones': [25, 45, 65]}

    def step_decay(epoch):
        lrate =training_params['optimizer']['lr']
        for milestone in training_params['lr_schedule']['milestones']:
            if epoch >= milestone:
                lrate = training_params['lr_schedule']['gamma']*lrate
        return lrate

    callbacks = [LearningRateScheduler(step_decay, verbose=0)]
else:
    callbacks=[]

def mse_cosine_loss(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) +  0.1*cosine_proximity(y_true, y_pred)

training_params['loss_weighting'] = 'mse:  1, cos_prox: 0.1'

def magnitude_weighted_loss(y_true, y_pred):
    errors  = abs(y_true - y_pred)
    weighted_errors = [(abs(y_true[i])+0.1) * error for i, error in enumerate(errors)]
    weighted_mse = np.mean(np.square(weighted_errors))
    return weighted_mse

loss = mse_cosine_loss
training_params['loss'] = 'mse_cosine_loss'

#################################################### model ##########################################################
if not (~os.path.exists(config_file.caffenet_models_weights)):
    download_network_weights()
else:
    print("pre-trained matconvnet model is already downloaded")






#################################################### callbacks #########################################################

# if(config_file.encoder_tenosrboard_logs is not None):
#     log_path = config_file.encoder_tenosrboard_logs
#     tb_callback = TensorBoard(log_path)
#     tb_callback.set_model(vision_model)
#     callbacks.append(tb_callback)

if save:
    if not os.path.exists('trained_encoders/' + model_name):
        os.makedirs('trained_encoders/' + model_name)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join('trained_encoders', model_name, 'best_ckpt.hdf5'),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    callbacks.append(model_checkpoint_callback)

#################################################### train######################################################
training_params['dropout'] = 0.5
enc_param = encoder_param(NUM_VOXELS)
enc_param.resolution = 28
enc_param.drop_out = training_params['dropout']
enc_param.image_channels = 1

training_params['conv_ch'] = 192
enc_param.conv_ch = training_params['conv_ch']

vision_model = encoder(enc_param)
#vision_model = encoder3(enc_param)
vision_model.compile(loss=loss, optimizer=optimizer,metrics=[mse_cosine_loss, 'mse','cosine_proximity','mae'])
print(vision_model.summary())
training_params['max_shift'] = 5
train_generator = batch_generator_enc(x_train, y_train, batch_size=training_params['batch_size'],max_shift = training_params['max_shift'])
test_generator = batch_generator_enc(x_test, y_test, batch_size=training_params['val_batch_size'],max_shift = 0)

training = vision_model.fit_generator(
    train_generator, 
    validation_data=test_generator, 
    validation_steps=1, 
    epochs=training_params['epochs'],
    verbose=2,
    use_multiprocessing=False,
    callbacks=callbacks
    ) 


############################################ Save ####################################################

# Added by Shawn Carere
testing = vision_model.evaluate_generator(test_generator)
testing_dict = {vision_model.metrics_names[i] : testing[i] for i in range(len(testing))}
# print('Loss: ', vision_model.loss)
# print('Optimizer: ', vision_model.optimizer)
# print('Metrics: ', vision_model.metrics)

# Save
if save:
    if not os.path.exists('trained_encoders/' + model_name):
        os.makedirs('trained_encoders/' + model_name)

    vision_model.save_weights('trained_encoders/' + model_name + '/model_weights.hdf5')
    fig, ax = plt.subplots()
    ax.plot(training.history['loss'])
    ax.plot(training.history['val_loss'])
    ax.legend(['train_loss', 'val_loss'])
    plt.savefig('trained_encoders/' + model_name + '/val_loss.png')
    fig, ax = plt.subplots()
    ax.plot(training.history['mse_cosine_loss'])
    ax.plot(training.history['val_mse_cosine_loss'])
    ax.legend(['train_loss', 'val_loss'])
    plt.savefig('trained_encoders/' + model_name + '/val_loss_no_reg.png')
    fig, ax = plt.subplots()
    ax.plot(training.history['loss'])
    ax.plot(training.history['val_loss'])
    ax.legend(['train_loss', 'val_loss'])
    ax.set_ylim(top=1.2)
    plt.savefig('trained_encoders/' + model_name + '/val_loss_zoom.png')
    # with open('trained_models/' + model_name + '/config.yaml', 'w') as file:
    #     yaml.dump(vision_model.get_config(), file) # Save archetecture

    with open('trained_encoders/' + model_name + '/train_config.yaml', 'w') as file:
        yaml.dump(training_params, file)
    
    with open('trained_encoders/' + model_name + '/test_history.json', 'w') as file:
        json.dump(testing_dict, file)

    hist_df = pd.DataFrame(training.history)
    # convert lr to float64
    if 'lr' in training.history.keys():
        training.history['lr'] = [np.float64(lr) for lr in training.history['lr']]

    with open('trained_encoders/' + model_name + '/train_history.json', 'w') as file:
        json.dump(training.history, file) # save training history