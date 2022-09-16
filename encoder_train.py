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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy

# OG Imports
import copy
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint, Callback
from keras.backend.tensorflow_backend import set_session, clear_session
from keras.losses import mean_squared_error, cosine_proximity, mean_squared_logarithmic_error
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

os.environ["CUDA_VISIBLE_DEVICES"] = config_file.GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
set_session(tf.Session(config=gpu_config))

# Set script params
save = True
model_name = 'encoder_neg_vox_redo'
training_params = {}
training_params['batch_size'] = 64
training_params['val_batch_size'] = 50
training_params['epochs'] = 80
training_params['roi'] = 'ROI_VC'
training_params['scale_type'] = 'standard'
training_params['outlier_removal'] = 0
training_params['normalization'] = 1

# loss = mean_squared_logarithmic_error
# training_params['loss'] = 'msle'

initial_lrate = 0.1
optimizer = SGD(lr=initial_lrate, decay = 0.0 , momentum = 0.9,nesterov=True)
training_params['optimizer'] = optimizer.get_config()

#################################################### data load ##########################################################
print('Script: encoder_train.py')
handler = data_handler(matlab_file = config_file.kamitani_data_mat, test_img_csv=config_file.test_image_ids, train_img_csv=config_file.train_image_ids)
handler.print_meta_desc()
Y,Y_test,Y_test_avg = handler.get_data(
    roi = training_params['roi'], 
    normalize=training_params['normalization'],
    scale=training_params['scale_type'], 
    remove_outliers=training_params['outlier_removal']
    )

labels_train, labels = handler.get_labels()


file= np.load(config_file.images_npz)
X = file['train_images']
X_test_avg = file['test_images']

X= X[labels_train]
x_train = X
X_test_all = X_test_avg[labels]
x_test = X_test_avg

corr = np.loadtxt(os.path.join(config_file.project_dir, 'beliy_vox_corr.csv')).astype(float)
idx = np.where(corr < 0)[0]
y_train = Y[:, idx]
y_test = Y_test_avg[:, idx]
NUM_VOXELS = y_train.shape[1]
print('y_train shape: ', y_train.shape)

print('Model Name: ', model_name)

#################################################### losses & schedule ##########################################################

# default lr milestones should be 20, 35, 45, 50 startinng at 0.1 lr and gamma of 0.1

training_params['lr_schedule'] = True
if training_params['lr_schedule']:
    training_params['lr_schedule'] = {
        'gamma':0.1,
        'milestones': [20, 35, 45, 50]}

    def step_decay(epoch):
        lrate =training_params['optimizer']['lr']
        for milestone in training_params['lr_schedule']['milestones']:
            if epoch >= milestone:
                lrate = training_params['lr_schedule']['gamma']*lrate
        return lrate

    callbacks = [LearningRateScheduler(step_decay, verbose=0)]
else:
    callbacks=[]

def combined_loss(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) +  0.1*cosine_proximity(y_true, y_pred)

def magnitude_weighted_loss(y_true, y_pred):
    errors  = abs(y_true - y_pred)
    weighted_errors = [(abs(y_true[i])+0.1) * error for i, error in enumerate(errors)]
    weighted_mse = np.mean(np.square(weighted_errors))
    return weighted_mse

loss = combined_loss
training_params['loss'] = 'combined_loss'

#################################################### model ##########################################################
if not (~os.path.exists(config_file.caffenet_models_weights)):
    download_network_weights()
else:
    print("pre-trained matconvnet model is already downloaded")

enc_param = encoder_param(NUM_VOXELS)
enc_param.drop_out = 0.5

vision_model = encoder(enc_param)
vision_model.compile(loss=loss, optimizer=optimizer,metrics=['mse','cosine_proximity','mae', combined_loss])
print(vision_model.summary())

#################################################### callbacks #########################################################

if(config_file.encoder_tenosrboard_logs is not None):
    log_path = config_file.encoder_tenosrboard_logs
    tb_callback = TensorBoard(log_path)
    tb_callback.set_model(vision_model)
    callbacks.append(tb_callback)

if save:
    if not os.path.exists('trained_encoders/' + model_name):
        os.makedirs('trained_encoders/' + model_name)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join('trained_encoders', model_name, 'best_ckpt.hdf5'),
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Custom callback for saving every batch
    # class CustomCallback(Callback):
    #     def __init__(self, filepath, save_freq, **kwargs):
    #         self.model_name = filepath
    #         self.save_freq = save_freq
    #         super().__init__(**kwargs)
    #         self.step = 0
                
    #     def on_batch_end(self, batch, logs=None):
    #         #filename = self.model_name + "_batch_" + f"{batch+1:02}"  + '.hdf5'
    #         filename = self.model_name + "_step_" + f"{self.step+1:02}"  + '.hdf5'
    #         self.step = self.step + 1
    #         self.model.save_weights(filename)
    #         print("\nsaved checkpoint: " + filename + "\n")
                

    # model_checkpoint_callback = CustomCallback(
    #     filepath=os.path.join('trained_encoders', model_name, 'ckpt'),
    #     save_freq=1,
    # )
    callbacks.append(model_checkpoint_callback)

#################################################### train######################################################
train_generator = batch_generator_enc(x_train, y_train, batch_size=training_params['batch_size'],max_shift = 5)
test_generator = batch_generator_enc(x_test, y_test, batch_size=training_params['val_batch_size'],max_shift = 0)

training = vision_model.fit_generator(train_generator, epochs=training_params['epochs'],validation_data=test_generator ,verbose=2,use_multiprocessing=False,callbacks=callbacks)

############################################ Save ####################################################

# Added by Shawn Carere
testing = vision_model.evaluate(x=x_test, y=y_test)
testing_dict = {vision_model.metrics_names[i] : testing[i] for i in range(len(testing))}
print('Loss: ', vision_model.loss)
print('Optimizer: ', vision_model.optimizer)
print('Metrics: ', vision_model.metrics)

# Save
if save:
    if not os.path.exists('trained_encoders/' + model_name):
        os.makedirs('trained_encoders/' + model_name)

    vision_model.save_weights('trained_encoders/' + model_name + '/model_weights.hdf5')

    # with open('trained_models/' + model_name + '/config.yaml', 'w') as file:
    #     yaml.dump(vision_model.get_config(), file) # Save archetecture

    with open('trained_encoders/' + model_name + '/train_config.yaml', 'w') as file:
        yaml.dump(training_params, file)
    
    with open('trained_encoders/' + model_name + '/test_history.json', 'w') as file:
        json.dump(testing_dict, file)

    if 'lr' in training.history.keys():
        training.history['lr'] = [np.float64(lr) for lr in training.history['lr']]

    with open('trained_encoders/' + model_name + '/train_history.json', 'w') as file:
        json.dump(training.history, file) # save training history

    fig, ax = plt.subplots()
    ax.plot(training.history['loss'])
    ax.plot(training.history['val_loss'])
    ax.legend(['train_loss', 'val_loss'])
    plt.savefig('trained_encoders/' + model_name + '/val_loss.png')
    fig, ax = plt.subplots()
    ax.plot(training.history['combined_loss'])
    ax.plot(training.history['val_combined_loss'])
    ax.legend(['train_loss', 'val_loss'])
    plt.savefig('trained_encoders/' + model_name + '/val_loss_no_reg.png')

    