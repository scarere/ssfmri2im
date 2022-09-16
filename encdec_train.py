
import os
import config_file
import yaml
import json
from scipy import stats
import pandas as pd
from helper_functions import *
os.environ["CUDA_VISIBLE_DEVICES"] = config_file.GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
from keras.callbacks import TensorBoard, LearningRateScheduler

from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from Utils.misc import calc_snr, log_image_collage_callback
from Utils.image_functions import *

from Utils.image_loss import *
from Utils.batch_generator import *
from Models.models import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Settings
save=True
training_params = {}
training_params['encoder'] = 'encoder_base'
training_params['decoder_name'] = 'decoder__corr0.24_r0.2_50epoch'

epochs_drop =  10.0
epochs = int(epochs_drop*5)
training_params['epochs'] = epochs
training_params['batch_size_paired'] = 48
training_params['batch_size_unpaired'] = 16

base_dir = '/cluster/projects/uludag/shawn/ssfmri2im/'
#base_dir = ''
vgg_weights_path = base_dir + 'Models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
encoder_weights = base_dir + 'trained_encoders/' + training_params['encoder'] + '/model_weights.hdf5'

#################################################### data load #########################################################
handler = data_handler(matlab_file = config_file.kamitani_data_mat, test_img_csv=config_file.test_image_ids, train_img_csv=config_file.train_image_ids)
Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0)
labels_train, labels = handler.get_labels(imag_data = 0)

file= np.load(config_file.images_npz) #_56
X = file['train_images']
X_test_avg = file['test_images']

X= X[labels_train]
X_test = X_test_avg[labels]

NUM_VOXELS = Y.shape[1]

xyz, _ = handler.get_voxel_loc()
Y_all_voxels = Y
Y, Y_test_avg, _ = remove_low_variance_voxels(Y_all_voxels, Y_test_avg, xyz, threshold=1.0)
_, Y_test, xyz = remove_low_variance_voxels(Y_all_voxels, Y_test, xyz, threshold=1.0) # for calculating SNR
NUM_VOXELS = Y.shape[1]
#################################################### losses ##########################################################

snr  = calc_snr(Y_test,Y_test_avg,labels)
snr = snr/snr.mean()
SNR  = tf.constant(snr,shape = [1,len(snr)],dtype = tf.float32)

def mse_vox(y_true, y_pred):
    return K.mean(SNR*K.square(y_true-y_pred),axis=-1)

def mae_vox(y_true, y_pred):
    return K.mean(SNR*K.abs(y_true-y_pred),axis=-1)

def combined_voxel_loss(y_true, y_pred):
    return mae_vox(y_true, y_pred) +  0.1 *cosine_proximity(y_true, y_pred)

def maelog_vox(y_true, y_pred):
    return K.mean(SNR*K.log(K.abs(y_true-y_pred)+1),axis=-1)


Tv_reg =1

image_loss_ = image_loss(vgg_weights=vgg_weights_path)

def feature_loss(y_true, y_pred ):
    return 0.15*image_loss_.vgg_loss(y_true, y_pred,'block2_conv2')+0.7*image_loss_.vgg_loss(y_true, y_pred,'block1_conv2')+0.15*image_loss_.pixel_loss(y_true, y_pred)
    #return image_loss_.pixel_loss(y_true, y_pred)#image_loss_.vgg_loss(y_true, y_pred,'block1_conv2')

def combined_loss(y_true, y_pred):
    return feature_loss(y_true, y_pred)+  Tv_reg *total_variation_loss(y_pred)

#################################################### learning param & schedule #########################################


initial_lrate = 0.001

RESOLUTION = config_file.image_size

# I don't think any of these variables are actually used
examples = 16
include_decenc= 1
frac = 3

def step_decay(epoch):

   drop = 0.2
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

##################################################### model ############################################################
image_loss_.calc_norm_factors(X)

#encoder_weights = config_file.encoder_weights

# Load encoder
enc_param = encoder_param(NUM_VOXELS)
enc_param.drop_out = 0.25
encoder_model = encoder(enc_param)
encoder_model.trainable = False
encoder_model.load_weights(encoder_weights)
print(encoder_model.summary())

# select voxel subset
fmri_pred = encoder_model.predict(X_test_avg)
voxel_corr = []
vox_range = []
for i in range(NUM_VOXELS):
    voxel_corr.append(stats.pearsonr(fmri_pred[:, i], Y_test_avg[:, i])[0])
    vox_range.append(np.max(fmri_pred[:, i]) - np.min(fmri_pred[:, i]))

# convert to numpy arrays for easy voxel selection
voxel_corr = np.array(voxel_corr)
vox_range = np.array(vox_range)

# remove voxels with low correlations
hc_voxels = np.where(voxel_corr > 0.25)[0]
hr_voxels = np.where(vox_range > 0.2)[0]
hchr_voxels = np.intersect1d(hc_voxels, hr_voxels)

# Create decoder model
dec_param = decoder_param(NUM_VOXELS)
decoder_model = decoder_subset(dec_param, voxel_subset=hchr_voxels)
#decoder_model = decoder(dec_param)
print(decoder_model.summary())

#model = encdec_vox_subset(NUM_VOXELS, RESOLUTION, encoder_model, decoder_model, hc_voxels)
model = encdec(NUM_VOXELS,RESOLUTION,encoder_model,decoder_model)
model.compile(loss= {'out_rec_img':combined_loss,'out_pred_voxel':combined_voxel_loss},loss_weights=[1.0,1.0],optimizer=Adam(lr=5e-4,amsgrad=True),metrics={'out_rec_img':['mse','mae']})
print(model.summary())
##################################################### callbacks ########################################################
callback_list = []

if(config_file.decoder_tenosrboard_logs is not None):
    callback = TensorBoard(config_file.decoder_tenosrboard_logs)
    callback.set_model(model)
    callback_list.append(callback)


reduce_lr = LearningRateScheduler(step_decay)
callback_list.append(reduce_lr)
if not os.path.exists(config_file.results):
    os.makedirs(config_file.results)

callback_list.append( log_image_collage_callback(Y_test_avg, X_test_avg, decoder_model, dir = config_file.results+'/test_collge_ep/'))
callback_list.append( log_image_collage_callback(Y[0:50], X[0:50], decoder_model, dir = config_file.results+'/train_collge_ep/'))
##################################################### generators #######################################################

loader_train = batch_generator_encdec(X, Y, Y_test, labels, batch_paired = 48, batch_unpaired = 16)
loader_test = batch_generator_encdec(X_test_avg, Y_test_avg, Y_test, labels, batch_paired = 50, batch_unpaired = 0)
##################################################### fit & save #######################################################

training = model.fit_generator(loader_train, epochs=epochs, validation_data=loader_test, validation_steps=1, verbose=2,callbacks=callback_list,workers=5,use_multiprocessing=True) #epochs
# image_collage([X_test_avg,decoder_model.predict(Y_test_avg)], rows =10, border =5,save_file = config_file.results+'/collage.jpeg')
# save_images(decoder_model.predict(Y_test_avg),images_orig = X_test_avg ,folder=config_file.results+'/test/')
# save_images(decoder_model.predict(Y[0:50]),images_orig = X[0:50] ,folder=config_file.results+'/train/')


# if(config_file.decoder_weights is not None):
#     decoder_model.save_weights(config_file.decoder_weights)

# Save

testing = model.evaluate_generator(loader_test, verbose=2, workers=5, use_multiprocessing=True, )
print(type(testing))
print(testing)
testing_dict = {model.metrics_names[i] : testing[i] for i in range(len(testing))}

if save:
    if not os.path.exists('trained_decoders/' + training_params['decoder_name']):
        os.makedirs('trained_decoders/' + training_params['decoder_name'])

    image_collage([X_test_avg,decoder_model.predict(Y_test_avg)], rows =10, border =5,save_file = 'trained_decoders/' + training_params['decoder_name'] + '/collage.jpeg')
    save_images(decoder_model.predict(Y_test_avg),images_orig = X_test_avg ,folder='trained_decoders/' + training_params['decoder_name'] + '/test/')
    save_images(decoder_model.predict(Y[0:50]),images_orig = X[0:50] ,folder='trained_decoders/' + training_params['decoder_name'] + '/train/')
        
    decoder_model.save_weights('trained_decoders/' + training_params['decoder_name'] + '/model_weights.hdf5')

    with open('trained_decoders/' + training_params['decoder_name'] + '/train_config.yaml', 'w') as file:
        yaml.dump(training_params, file)
    
    with open('trained_decoders/' + training_params['decoder_name'] + '/test_history.json', 'w') as file:
        json.dump(testing_dict, file)

    hist_df = pd.DataFrame(training.history)

    with open('trained_decoders/' + training_params['decoder_name'] + '/train_history_df.csv', 'w') as file:
        hist_df.to_csv(file)

    with open('trained_decoders/' + training_params['decoder_name'] + '/train_history.json', 'w') as file:
        json.dump(training.history, file) # save training history