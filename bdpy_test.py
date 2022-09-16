import bdpy
from bdpy.mri.image import export_brain_image
import numpy as np
import nibabel as nib
import time

ts = time.time()
data = bdpy.BData('KamitaniData/fmri/Subject3.mat')
data.show_metadata() # Prints the metadata out to the console
voxel_data = data.select('ROI_VC') # Returns the voxel data with shape (sample, num_voxels)
print(voxel_data.shape)

# bdpy function could be improved to not require a template when given xyz coords,
x = data.get_metadata('voxel_x', where='VoxelData')
y = data.get_metadata('voxel_y', where='VoxelData')
z = data.get_metadata('voxel_z', where='VoxelData')
xyz = np.vstack((x, y, z))  # (3, num_voxels)

tc = time.time() - ts
print(tc)
# attempt to use preproc anat image as template
nifti_im = export_brain_image(voxel_data[0], template='KamitaniData/func_raw/sub-03_ses-perceptionTest01_task-perception_run-01_bold_preproc.nii.gz', xyz=xyz)
tc = time.time() - ts
print(tc)
nib.save(nifti_im, 'sub03_sample0.nii.gz')