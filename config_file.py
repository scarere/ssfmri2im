
import os
GPU_ID = "0"

#####################  PATHS  ######################################
imagenet_dir = "/Users/scarere/Documents/UofT/OneDrive-UofT/ssfmri2im/imagenet"
imagenet_wind_dir = os.path.join(imagenet_dir,"KamitaniImages/")
external_images_dir =  os.path.join(imagenet_dir,"ImageNetVal2011")

project_dir = "/Users/scarere/Documents/UofT/OneDrive-UofT/ssfmri2im/"
images_npz = os.path.join(project_dir,"KamitaniData/images/images_112.npz")
kamitani_data_format = True
kamitani_data_mat = os.path.join(project_dir,"KamitaniData/fmri/Subject3.mat")
test_image_ids = os.path.join(project_dir,"KamitaniData/images/image_test_id.csv")
train_image_ids = os.path.join(project_dir,"KamitaniData/images/image_training_id.csv")
caffenet_models_weights = os.path.join(project_dir,"models/imagenet-caffe-ref.mat")
results  = os.path.join(project_dir,"results/")


encoder_weights = os.path.join(project_dir,"models/encoder.hdf5")
retrain_encoder = True
decoder_weights = None

encoder_tenosrboard_logs = None
decoder_tenosrboard_logs = None


#####################  pretrained mat conv net weights (alexnet)  ######################################

DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-caffe-ref.mat'
FILENAME = 'imagenet-caffe-ref.mat'
EXPECTED_BYTES = 228031200

##################### PARAMS ######################################

image_size = 112
