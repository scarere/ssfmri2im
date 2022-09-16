import numpy as np
from numpy.lib.function_base import percentile
from scipy import stats
from tensorflow.python.ops.gen_math_ops import neg
import tensorflow as tf

def filter_by_max(train_data, test_data, xyz, threshold):
    '''
    Filter data by it's max amplitude to remove voxels that are always
    near zero

    Args:
        train_data: data to filter (n_samples, n_voxels), filtered based
            on train data
        test_data: test data to be filtered
        xyz: voxel coordinates

    Returns:
        train_filt: filtered train data with select voxels removed based
            on their max amplitude in train set
        test_filt: filtered test data with same select voxels removed
        xyz_filt: coordinates of voxels that remain
    '''

    max_amp = np.max(np.abs(train_data), axis=0)
    bool_filter = max_amp > threshold
    train_filt = train_data[:, bool_filter]
    test_filt = test_data[:, bool_filter]
    xyz_filt = xyz[:, bool_filter]
    return train_filt, test_filt, xyz_filt

def remove_low_variance_voxels(fmri_set1, fmri_set2, xyz, threshold):
    '''Removes voxels with low variance from the data

    Calculates the variance for each voxel using the firsts set of data and uses
    a threshold to remove voxels based on their variance in both datasets

    Args:
        fmri_set1: fmri data from which variance will be calculated for threshold. [samples, voxels]
        fmri_set2: fmri data to remove corresponding voxels from. [samples, voxels]
        xyz: the coordinates for each voxel. Should had shape [num_dim, num_voxels]
        threshold (int): The threshold variance. Voxels with variance below
            this value will be removed

    returns:
        fmri_set1_new: fmri data with low variance voxels removed
        fmri_set2_new: fmri data with the corresponding voxels removed
        xyz_new: The coordinates of the high variance voxels. 
    '''

    var = np.var(fmri_set1, axis=0)
    hv_voxels = var > threshold # Get indices of where high variance voxels are
    fmri_set1_new = fmri_set1[:, hv_voxels]
    fmri_set2_new = fmri_set2[:, hv_voxels]
    xyz_new = xyz[:, hv_voxels]
    
    return fmri_set1_new, fmri_set2_new, xyz_new

def remove_low_corr_voxels_v2(train_fmri, test_fmri, xyz,  threshold, pred_path):
    '''
    Removes voxels that have a low predictability based on voxel correlation between
    predictions of a previously trained encoder and ground truth activations. Uses a
    validation set to select voxels

    Args:
        train_fmri: training fmri data
        test_fmri: test fmri data
        xyz: voxel coordinates for nifti reconstruction
        threshold: minimum correlation
        pred_path: path to csv file containing predictionf from trained encoder 
            used to calculate correlation coefficients

    Returns
        train_fmri_hc: training fmri data for only high correlation voxels
        val_fmri_hc: the high correlation voxels of the data that was used
            select the high correlation voxels
        test_fmri_hc: test fmri data for only high correlation voxels. This data
            was not used to select the high correlation voxels.
        xyz_hc: coordinates for new set of voxels
    '''

    y_pred = np.loadtxt(pred_path, delimiter=',')
    num_samples = np.shape(y_pred)[0]
    val_fmri = test_fmri[:int(num_samples*0.8), :]
    y_pred = y_pred[:int(num_samples*0.8), ] # Only use validation set to select voxels
    NUM_VOXELS = y_pred.shape[1]
    voxel_corr = []
    for i in range(NUM_VOXELS):
        voxel_corr.append(stats.pearsonr(y_pred[:, i], val_fmri[:, i])[0])

    voxel_corr = np.array(voxel_corr)
    hc_voxels = np.where(voxel_corr > threshold)[0]
    train_fmri_hc = train_fmri[:, hc_voxels]
    val_fmri_hc = val_fmri[:, hc_voxels]
    test_fmri_hc = test_fmri[int(num_samples*0.8):, hc_voxels]
    xyz_hc = np.array(xyz)
    xyz_hc = xyz_hc[:, hc_voxels]
    
    return train_fmri_hc, val_fmri_hc, test_fmri_hc, xyz_hc

def remove_low_corr_voxels_v1(train_fmri, test_fmri, xyz,  threshold, pred_path):
    '''
    Removes voxels that have a low predictability based on voxel correlation between
    predictions of a previously trained encoder and ground truth activations. Uses
    the test set to select voxels, does not use a validation set.

    Args:
        train_fmri: training fmri data
        test_fmri: test fmri data
        xyz: voxel coordinates for nifti reconstruction
        threshold: minimum correlation
        pred_path: path to csv file containing predictionf from trained encoder 
            used to calculate correlation coefficients

    Returns
        train_fmri_hc: training fmri data for only high correlation voxels
        test_fmri_hc: test fmri data for only high correlation voxels. This data
            was not used to select the high correlation voxels.
        xyz_hc: coordinates for new set of voxels
    '''

    y_pred = np.loadtxt(pred_path, delimiter=',')
    NUM_VOXELS = y_pred.shape[1]
    voxel_corr = []
    for i in range(NUM_VOXELS):
        voxel_corr.append(stats.pearsonr(y_pred[:, i], test_fmri[:, i])[0])

    voxel_corr = np.array(voxel_corr)
    hc_voxels = np.where(voxel_corr > threshold)[0]
    train_fmri_hc = train_fmri[:, hc_voxels]
    test_fmri_hc = test_fmri[:, hc_voxels]
    xyz_hc = np.array(xyz)
    xyz_hc = xyz_hc[:, hc_voxels]
    
    return train_fmri_hc, test_fmri_hc, xyz_hc

def get_percentile_bounds(train_data, n):
    '''Returns a list of boundaries for quantization based on percentile.

    Args:
        train_data: The training data
        n (int): The number of bins to split the data up into. Must be 
            an odd number so that there can be a bin centered on zero

    Returns:
        bounds (list): A list of boundaries
    '''
    assert n%2 == 1, 'n argument must be an odd number'
    p = float(100/n) # The bin size

    # must double bin size for all bins except central bin since we are using magnitude
    p2 = 2*p
    mags = abs(np.array(train_data))
    
    bounds = [np.percentile(mags, p)] # Add bound for first bin
    iter = int((n-1)/2 - 1) # Subtract 1 since we already added first bound
    percentiles = [p]
    for i in range(iter):
        q = p + (i+1)*p2 # Add 2 to i since we want to start at 2 (we already added first bound for index 1)
        percentiles.append(q)
        bound = np.percentile(mags, q=q)
        bounds.append(bound)

    print(bounds)
    neg_bounds = np.multiply(bounds, -1)
    return np.concatenate([np.flip(neg_bounds), bounds]).tolist(), percentiles

def pearson_r_approx(y_true, y_pred, axis=0):
    '''Calculates an approximation of the correlation between between model 
    predictions and ground truth using the cosine similarity function.

    Args:
        y_pred (Tensor): The model predictions. A tensor with shape 
            (batch_size, D) where D is the dimensionality of the output vector
        y_test (Tensor): The ground truth values. A tensor with shape 
            (batch_size, D) where D is the dimensionality of the output vector
        axis (int): Which axis reduce when computing the correlations. If 0, 
            computes correlations between vectors of length (batch_size). If 1, 
            computes correlations between vectors of length (D)

    Returns:
        corrs (Tensor): A tensor containing all the correlation coefficients for
            the batch. If axis is 0, corrs shape will be (D,), if axis is 1 corrs
            shape will be (batch_size,)
    '''
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=axis, keepdims=True)
    my = tf.reduce_mean(y, axis=axis, keepdims=True)
    xm, ym = x - mx, y - my
    t1_norm = tf.nn.l2_normalize(xm, axis = axis)
    t2_norm = tf.nn.l2_normalize(ym, axis = axis)
    cosine = -1*tf.losses.cosine_similarity(t1_norm, t2_norm, axis = axis)
    return cosine

def pearson_r(y_true, y_pred, axis=0):
    '''Calculates the exact correlation between between model predictions and 
    ground truth

    Args:
        y_pred (Tensor): The model predictions. A tensor with shape 
            (batch_size, D) where D is the dimensionality of the output vector
        y_test (Tensor): The ground truth values. A tensor with shape 
            (batch_size, D) where D is the dimensionality of the output vector
        axis (int): Which axis reduce when computing the correlations. If 0, 
            computes correlations between vectors of length (batch_size). If 1, 
            computes correlations between vectors of length (D)

    Returns:
        corrs (Tensor): A tensor containing all the correlation coefficients for
            the batch. If axis is 0, corrs shape will be (D,), if axis is 1 corrs
            shape will be (batch_size,)
    '''
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=axis, keepdims=True)
    my = tf.reduce_mean(y, axis=axis, keepdims=True)
    xm, ym = x - mx, y - my
    numerator = tf.reduce_sum(tf.multiply(xm, ym), axis=axis)
    denominator = tf.sqrt(tf.multiply(tf.reduce_sum(xm**2, axis=axis), tf.reduce_sum(ym**2, axis=axis)))
    return tf.divide(numerator, denominator)

def mutual_info(x, y):
    '''Calculates the mutual info between to random variables X and Y.

    Args:
        x (n_samples): Observations of X
        y (n_samples): Observations of Y

    Returns
        m (float): Mutual info score between X and Y
    '''

    kx = stats.gaussian_kde(x)
    ky = stats.gaussian_kde(y)
    xy = np.vstack([x, y])
    kxy = stats.gaussian_kde(xy)

    px = kx(x)
    py = ky(y)
    X, Y = np.meshgrid(x, y)
    XY = np.vstack([X.ravel(), Y.ravel()])
    pxy = np.reshape(kxy(XY).T, X.shape)
    pxpy = np.matmul(np.expand_dims(px, axis=1), np.expand_dims(py, axis=0)) # rows select x, cols select y
    # since we are using natural log, the mutual info returned is measured in 'nats'
    m = np.sum(np.multiply(pxy, np.log(np.divide(pxy, pxpy))))
    return m

def norm_mutual_info(x, y):
    kx = stats.gaussian_kde(x)
    ky = stats.gaussian_kde(y)
    xy = np.vstack([x, y])
    kxy = stats.gaussian_kde(xy)

    px = kx(x)
    py = ky(y)
    X, Y = np.meshgrid(x, y)
    XY = np.vstack([X.ravel(), Y.ravel()])
    pxy = np.reshape(kxy(XY).T, X.shape)
    pxpy = np.matmul(np.expand_dims(px, axis=1), np.expand_dims(py, axis=0)) # rows select x, cols select y
    # since we are using natural log, the mutual info returned is measured in 'nats'
    m = np.sum(np.multiply(pxy, np.log(np.divide(pxy, pxpy))))
    hx = stats.entropy(px)
    hy = stats.entropy(py)
    return m/(hx + hy)
    