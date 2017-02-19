"""
    Main Satellite training a testing code.
    version: submission for sub_ep30_train12800_unet320_autoscaleTo320x6_v2.csv.csv, lb score = 0.26846
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import shapely.wkt
import shapely.affinity
import pickle
import os
import math
import jaccard_tools
import config
import image_source
import polygon_tools

N_Cls = 10
ISZ = 320  # unet input size, 160 baseline

DF = pd.read_csv(config.glb_base_dir + '/train_wkt_v4.csv')
GS = pd.read_csv(config.glb_base_dir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(os.path.join(config.glb_base_dir, 'sample_submission.csv'))


IMAGE_DEPTH = 3  # 8 for image M, 20 for all, 3 for 3-band

NET_SIZE_MULT = 8  # 6 is baseline
NET_FIELD_SIZE = ISZ * NET_SIZE_MULT
MAX_IMAGE_SIZE = 3403
MIN_IMAGE_SIZE = 3335
IMAGE_SF = NET_FIELD_SIZE / MAX_IMAGE_SIZE
IMAGE_MIN_SIZE = int(IMAGE_SF * MIN_IMAGE_SIZE)  # 835 for image M, 930 for image 3
# IMAGE_MIN_SIZE = NET_FIELD_SIZE
print('SF', IMAGE_SF, 'IMAGE_MIN_SIZE', IMAGE_MIN_SIZE, 'NET_FILED_SIZE', NET_FIELD_SIZE)


class TrainingDB:
    def __init__(self):
        self.x = []
        self.y = []

    def build_training_db(self, image_sf, train_image_ids, num_class, subset_size=0):
        if subset_size == 0:
            train_image_id_list = train_image_ids
        else:
            train_image_id_list = train_image_ids[:subset_size]

        train_image_list = []
        train_mask_list = []
        for i, idx in enumerate(train_image_id_list):
            # get images for this image id
            img_mat = image_source.get_image_from_id(idx, image_sf)
            img_mat = image_source.stretch_n(img_mat)
            train_image_list.append(img_mat)
            image_mask = np.zeros((img_mat.shape[0], img_mat.shape[1], num_class), np.uint8)
            for z in range(num_class):
                image_mask[:, :, z] = polygon_tools.generate_mask_for_image_and_class(
                    (img_mat.shape[0], img_mat.shape[1]), idx, z + 1, GS, DF)
            train_mask_list.append(image_mask)
            print('{}/{}'.format(i+1, len(train_image_id_list)), idx, 'shapes:', img_mat.shape, image_mask.shape)
        self.x = train_image_list
        self.y = train_mask_list

    @staticmethod
    def get_training_mask(image_id, img_mat, num_class):
        image_mask = np.zeros((img_mat.shape[0], img_mat.shape[1], num_class), np.uint8)
        for z in range(num_class):
            image_mask[:, :, z] = polygon_tools.generate_mask_for_image_and_class(
                (img_mat.shape[0], img_mat.shape[1]), image_id, z + 1, GS, DF)
        return image_mask

    def get_random_patches(self, num_patch, patch_size, class_prob_thresh=None, aug=True):
        if class_prob_thresh is None:
            tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
        else:
            tr = class_prob_thresh

        x = []  # return images here
        y = []  # return masks here
        # get num_patch random patches (images and masks)
        while len(x) < num_patch:
            # pick a random image for the image list 'x'
            rand_image_index = random.randint(0, len(self.x)-1)
            img_mat = self.x[rand_image_index]
            msk_mat = self.y[rand_image_index]
            num_class = msk_mat.shape[2]
            # pick a sub-region of the image, smaller that the image to find random patches in.
            xm = img_mat.shape[0] - patch_size
            ym = img_mat.shape[1] - patch_size
            # get a random patch with the sub-region
            xc = random.randint(0, xm)
            yc = random.randint(0, ym)
            im = img_mat[xc:xc + patch_size, yc:yc + patch_size]
            ms = msk_mat[xc:xc + patch_size, yc:yc + patch_size]
            # check that the image contains enough labeled pixels
            # print('patch', patch_count, 'size', xm, ym, 'point', xc, yc, im.shape, ms.shape)
            for j in range(num_class):
                sm = float(np.sum(ms[:, :, j]))
                if (sm / (patch_size ** 2)) >= tr[j]:
                    if aug:
                        if random.uniform(0, 1) > 0.5:
                            im = im[::-1]
                            ms = ms[::-1]
                        if random.uniform(0, 1) > 0.5:
                            im = im[:, ::-1]
                            ms = ms[:, ::-1]
                    x.append(im)
                    y.append(ms)
                    # print(len(x), 'im shape', im.shape, 'ms shape', ms.shape)

        x = 2 * np.transpose(x, (0, 3, 1, 2)) - 1
        y = np.transpose(y, (0, 3, 1, 2))
        # print('x shape', x.shape, 'y shape', y.shape)

        return x[:num_patch, :, :, :], y[:num_patch, :, :, :]

    @staticmethod
    def print_mask_stats(mask_mat):
        n_class = mask_mat.shape[1]
        n_sample = mask_mat.shape[0] * mask_mat.shape[2] * mask_mat.shape[3]
        probs = np.zeros(n_class)
        print('------------------')
        for ic in range(n_class):
            sums = np.sum(mask_mat[:, ic, :, :])
            apriori_prob = sums / n_sample
            probs[ic] = apriori_prob
            print('class', ic, 'sum', sums, 'prob', apriori_prob)
        return probs

    def compute_balance_thresh(self, num_patch):
        num_class = 10
        for thresh in np.linspace(0.0, 1.0, 20):
            print('--------------------')
            print(thresh)
            x, y = self.get_random_patches(num_patch, ISZ, [thresh]*num_class)
            self.print_mask_stats(y)


def build_and_display_thumbs_from_image_matrix(image_mat):
    n_image = image_mat.shape[0]
    n_deep = image_mat.shape[1]  # assume all images are the same depth
    # find the max x and y dims
    max_rows = image_mat.shape[2]
    max_cols = image_mat.shape[3]
    n_blocks = math.ceil(math.sqrt(n_image))
    mosaic = np.zeros((max_rows*n_blocks, max_cols*n_blocks, n_deep))
    print(max_rows, max_cols, n_blocks, mosaic.shape)
    k = 0
    for i in range(n_blocks):
        for j in range(n_blocks):
            if k < n_image:
                ix = i * max_rows
                jx = j * max_cols
                img1 = image_mat[k, :, :, :]  # image_mat = (n_image, n_deep, img_rows, img_cols)
                img2 = np.transpose(img1, (1, 2, 0))  # img2 = (img_rows, img_cols, n_deep)
                mosaic[ix: ix+max_rows, jx:jx+max_cols, :] = img2
            k += 1
    # display it
    plt.imshow(mosaic)
    plt.show()


def build_and_display_thumbs_from_image_list(image_list: []):
    n_deep = image_list[0].shape[2]  # assume all images are the same depth
    # find the max x and y dims
    max_rows = max([image.shape[0] for image in image_list])
    max_cols = max([image.shape[1] for image in image_list])
    n_blocks = math.ceil(math.sqrt(len(image_list)))
    mosaic = np.zeros((max_rows*n_blocks, max_cols*n_blocks, n_deep))
    print(max_rows, max_cols, n_blocks, mosaic.shape)
    k = 0
    for i in range(n_blocks):
        for j in range(n_blocks):
            if k < len(image_list):
                ix = i * max_rows
                jx = j * max_cols
                n_rows = image_list[k].shape[0]
                n_cols = image_list[k].shape[1]
                mosaic[ix: ix+n_rows, jx:jx+n_cols, :] = image_list[k]
            k += 1
    # display it
    plt.imshow(mosaic)
    plt.show()


def get_unet_160():
    inputs = Input((IMAGE_DEPTH, ISZ, ISZ))

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(N_Cls, 1, 1, activation='sigmoid')(conv9)

    unet_model = Model(input=inputs, output=conv10)
    unet_model.compile(optimizer=Adam(), loss='binary_crossentropy',
                       metrics=[jaccard_tools.jaccard_coef, jaccard_tools.jaccard_coef_int, 'accuracy'])
    return unet_model


def get_unet():
    inputs = Input((IMAGE_DEPTH, ISZ, ISZ))

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv5a = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
    conv5a = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5a)

    up6a = merge([UpSampling2D(size=(2, 2))(conv5a), conv5], mode='concat', concat_axis=1)
    conv6a = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6a)
    conv6a = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6a)

    up6 = merge([UpSampling2D(size=(2, 2))(conv6a), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(N_Cls, 1, 1, activation='sigmoid')(conv9)

    unet_model = Model(input=inputs, output=conv10)
    unet_model.compile(optimizer=Adam(), loss='binary_crossentropy',
                       metrics=[jaccard_tools.jaccard_coef, jaccard_tools.jaccard_coef_int, 'accuracy'])
    return unet_model


def get_scalers(im_size, x_max, y_min):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def train_net(train_image_db: TrainingDB, train_set_size=6400, val_set_size=1000, num_epochs=1, train_batch_size=64):

    def fit_train_data_generator(train_gen_db, gen_batch_size):
        while True:
            x_trn, y_trn = train_gen_db.get_random_patches(gen_batch_size, ISZ)
            yield x_trn, y_trn

    print('*** train_net: using', num_epochs, 'epochs with', train_set_size, 'samples per epoch',
          'validation set size', val_set_size, 'number of epochs', num_epochs,
          'batch size', train_batch_size)

    train_model = get_unet()

    filepath = config.glb_base_dir + "/weights/" + "model-{epoch:02d}-{val_loss:.5f}.hdf5"
    model_checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=False, verbose=1)
    print('begin fit_generator...')
    train_model.fit_generator(fit_train_data_generator(train_image_db, train_batch_size),
                              samples_per_epoch=train_set_size, nb_epoch=num_epochs,
                              callbacks=[model_checkpoint], verbose=1,
                              validation_data=fit_train_data_generator(train_image_db, train_batch_size),
                              nb_val_samples=val_set_size)

    print('train_net model complete.')
    return train_model


def predict_id_nopad(image_id, pred_model, mask_thresh_vec, image_sf):
    img_mat = image_source.get_image_from_id(image_id, image_sf)
    x = image_source.stretch_n(img_mat)

    field_size = ISZ * NET_SIZE_MULT

    cnv = np.zeros((field_size, field_size, IMAGE_DEPTH)).astype(np.float32)
    prd = np.zeros((N_Cls, field_size, field_size)).astype(np.float32)
    cnv[:img_mat.shape[0], :img_mat.shape[1], :] = x

    for i in range(0, NET_SIZE_MULT):
        line = []
        for j in range(0, NET_SIZE_MULT):
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = pred_model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]

    # threshold prediction to create a binary mask
    for i in range(N_Cls):
        prd[i] = prd[i] > mask_thresh_vec[i]

    return prd[:, :img_mat.shape[0], :img_mat.shape[1]]


def predict_image_using_padding(img_mat, pred_model, pad_size):
    # get dimensions
    # print('img_mat shape', img_mat.shape, 'pad size', pad_size)
    rows = img_mat.shape[0]
    cols = img_mat.shape[1]
    depth = img_mat.shape[2]
    num_class = pred_model.output_shape[1]
    isz = pred_model.input_shape[2]
    new_isz = isz - 2 * pad_size
    delta = new_isz  # this is the size of the valid output of predict,

    new_rows = isz * math.ceil(rows / isz)
    new_cols = isz * math.ceil(cols / isz)

    # create arrays for input to predict
    conv = np.zeros((new_rows+new_isz, new_cols+new_isz, depth), np.float32)
    conv[pad_size:pad_size+rows, pad_size:pad_size+cols, :] = img_mat
    # print('conv shape', conv.shape)

    net_rows = math.ceil(new_rows / new_isz)
    net_cols = math.ceil(new_cols / new_isz)
    pred_rows = new_isz * net_rows
    pred_cols = new_isz * net_cols

    # create output image
    pred = np.zeros((num_class, pred_rows, pred_cols), np.float32)
    # print('pred shape', pred.shape)

    # create x to hold all sub-images for prediction
    k = 0
    istart = 0
    x = np.zeros((net_rows*net_cols, isz, isz, depth), np.float32)
    for i in range(0, net_rows):
        jstart = 0
        for j in range(0, net_cols):
            # print(i,j,'[',istart,':',istart+isz,',',jstart,':',jstart+isz,']')
            x[k, :, :, :] = conv[istart:istart+isz, jstart:jstart+isz, :]
            k += 1
            jstart += delta
        istart += delta

    # transpose all patches for prediction
    x_t = 2 * np.transpose(x, (0, 3, 1, 2)) - 1

    # predict patches
    # print('predict on x_t', x_t.shape)
    y = pred_model.predict(x_t, batch_size=16)

    # rebuild image from predicted patches
    k = 0
    istart = 0
    for i in range(0, net_rows):
        jstart = 0
        for j in range(0, net_cols):
            # print(i,j,'[',istart,':',istart+new_isz,',',jstart,':',jstart+new_isz,']')
            pred[:, istart:istart+new_isz, jstart:jstart+new_isz] = \
                y[k, :, pad_size:pad_size+new_isz, pad_size:pad_size+new_isz]
            k += 1
            jstart += delta
        istart += delta

    # return an image of the same size as the input
    img_prob = pred[:, :img_mat.shape[0], :img_mat.shape[1]]
    # print('img_prob shape', img_prob.shape)
    return img_prob


def map_image_to_probability_matrix(img_mat, pred_model):
    # x = image_source.stretch_n(img_mat)

    field_size = ISZ * NET_SIZE_MULT

    cnv = np.zeros((field_size, field_size, IMAGE_DEPTH)).astype(np.float32)
    prd = np.zeros((N_Cls, field_size, field_size)).astype(np.float32)
    cnv[:img_mat.shape[0], :img_mat.shape[1], :] = img_mat

    for i in range(0, NET_SIZE_MULT):
        line = []
        for j in range(0, NET_SIZE_MULT):
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = pred_model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]

    return prd[:, :img_mat.shape[0], :img_mat.shape[1]]


def threshold_prob_mask(prob_mat, mask_thresh_vec):
    prob_mask = np.zeros(prob_mat.shape, np.float32)
    for i in range(len(mask_thresh_vec)):
        prob_mask[i, :, :] = prob_mat[i, :, :] > mask_thresh_vec[i]
    prob_mask = prob_mask.astype(np.uint8)
    return prob_mask


def predict_and_write_masks(prediction_model, mask_thresh_vec, mask_dir, image_sf):
    print('*** predict and write masks...')
    img_list_len = len(set(SB['ImageId'].tolist()))
    for i, idx in enumerate(sorted(set(SB['ImageId'].tolist()))):
        img_mat = image_source.get_image_from_id(idx, image_sf)
        img_mat = image_source.stretch_n(img_mat)
        pred_prob = predict_image_using_padding(img_mat, prediction_model, pad_size=16)
        pred_mask = threshold_prob_mask(pred_prob, mask_thresh_vec)
        np.save(mask_dir + '/msk_%s' % idx, pred_mask)
        if i % 50 == 0:
            print(' id', i+1, 'of', img_list_len, '@id:', idx)


def make_submission_file_from_masks(mask_dir, sub_file_name):
    print('*** make submission file from masks...')
    df = pd.read_csv(os.path.join(config.glb_base_dir, 'sample_submission.csv'))
    print(df.head())
    last_img_id = ''
    img_msk_all = None
    for idx, row in df.iterrows():
        img_id = row[0]
        kls = row[1] - 1
        if img_id != last_img_id:
            img_msk_all = np.load(mask_dir + '/msk_%s.npy' % img_id)
            img_msk_all = img_msk_all.astype(np.float32)
            last_img_id = img_id

        img_msk = img_msk_all[kls]

        pred_polygons = polygon_tools.mask_to_polygons(img_msk, epsilon=5, min_area=1.0)

        x_max = GS.loc[GS['ImageId'] == img_id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == img_id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(img_msk.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))

        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        if idx % 500 == 0:
            print('  id', idx)

    print(df.head())
    print('Writing submission file...')
    df.to_csv(sub_file_name, index=False)
    print('submit file complete.')


def check_predict(image_id, feature_id, pred_model, thresh, image_sf):
    img_mat = image_source.get_image_from_id(image_id, image_sf)
    img_mat = image_source.stretch_n(img_mat)

    # gt_mask = TrainingDB.get_training_mask(image_id, img_mat, 10)

    pred_prob = predict_image_using_padding(img_mat, pred_model, pad_size=16)
    print('pred_prob shape', pred_prob.shape)

    pred_mask = threshold_prob_mask(pred_prob, thresh)

    plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('image ID:' + image_id)
    ax1.imshow(img_mat[:, :, :], cmap=plt.get_cmap('gist_ncar'))
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('predict bldg pixels')
    ax2.imshow(pred_mask[feature_id, :, :], cmap=plt.get_cmap('gray'))
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('predict bldg polygons')
    ax3.imshow(polygon_tools.mask_for_polygons(polygon_tools.mask_to_polygons(pred_mask[feature_id, :, :], epsilon=1),
                                               img_mat.shape[:2]), cmap=plt.get_cmap('gray'))
    # ax4 = plt.subplot(2, 2, 4)
    # ax4.set_title('ground truth')
    # tf_mask = gt_mask == pred_mask
    # print('error count', np.sum(gt_mask[:, :, feature_id] != pred_mask[feature_id, :, :]))
    # ax4.imshow(gt_mask[:, :, feature_id])

    plt.show()


def check_predict_2(prediction_model, mask_thresh_vec, image_id, feature_id, mask_dir, image_sf):
    img_mat = image_source.get_image_from_id(image_id, image_sf)
    img_mat = image_source.stretch_n(img_mat)

    pred_prob = predict_image_using_padding(img_mat, prediction_model, pad_size=16)
    pred_mask0 = threshold_prob_mask(pred_prob, mask_thresh_vec)
    np.save(mask_dir + '/msk_%s' % image_id, pred_mask0)

    pred_mask = np.load(mask_dir + '/msk_%s.npy' % image_id)
    print('pred_mask shape', pred_mask.shape)

    plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('image ID:' + image_id)
    ax1.imshow(img_mat[:, :, :], cmap=plt.get_cmap('gist_ncar'))

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('predict bldg pixels')
    ax2.imshow(pred_mask[feature_id, :, :], cmap=plt.get_cmap('gray'))

    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('predict bldg polygons')
    ax3.imshow(polygon_tools.mask_for_polygons(polygon_tools.mask_to_polygons(pred_mask[feature_id, :, :], epsilon=1),
                                               img_mat.shape[:2]), cmap=plt.get_cmap('gray'))
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('ground truth')
    # tf_mask = gt_mask == pred_mask
    # print('error count', np.sum(gt_mask[:, :, feature_id] != pred_mask[feature_id, :, :]))
    # ax4.imshow(gt_mask[:, :, feature_id])

    plt.show()


def calculate_optimum_binary_thresholds(train_model_db, train_model, calc_size=0):
    if calc_size == 0:
        calc_size = len(train_model_db.x)

    num_thresh = 10
    num_class = train_model_db.y[0].shape[2]
    best_jacc = np.zeros(num_class)
    best_thresh = np.zeros(num_class)

    for idx in range(calc_size):
        y_pred_3d = map_image_to_probability_matrix(train_model_db.x[idx], train_model)
        y_pred_3d = np.rollaxis(y_pred_3d, 0, 3)
        y_true_3d = train_model_db.y[idx]
        print(idx, 'y_pred shape', y_pred_3d.shape, 'y_true shape', y_true_3d.shape)

        for ic in range(num_class):
            y_pred = y_pred_3d[:, :, ic]
            y_true = y_true_3d[:, :, ic]
            for j in range(10):
                thresh = float(j) / num_thresh
                y_pred_thresh = y_pred > thresh
                jacc = jaccard_tools.jaccard_similarity_score(y_true, y_pred_thresh)
                if jacc > best_jacc[ic]:
                    best_jacc[ic] = jacc
                    best_thresh[ic] = thresh
                print(idx, ic, thresh, 'jacc', jacc, best_jacc[ic], best_thresh[ic])
    j_score = np.average(best_jacc)
    return j_score, best_thresh


if __name__ == '__main__':
    build_train_db = True
    do_training = False
    build_masks_and_submissions = True
    generate_jaccard = True
    display_training_data = False

    print('Begin v0.0.22')

    if build_train_db:
        image_db = TrainingDB()
        image_db.build_training_db(IMAGE_SF,  sorted(DF.ImageId.unique()), N_Cls, subset_size=0)
        if display_training_data:
            build_and_display_thumbs_from_image_list(image_db.x)
            build_and_display_thumbs_from_image_list([img[:, :, 0:3] for img in image_db.y])
            build_and_display_thumbs_from_image_list([img[:, :, 3:6] for img in image_db.y])
    else:
        image_db = None

    if do_training:
        model = train_net(image_db, train_set_size=16*1000, val_set_size=10000, num_epochs=50, train_batch_size=16)
        model.save(config.glb_base_dir + '/weights/model_now.modx')
        img_val, msk_val = image_db.get_random_patches(2500, ISZ)
        score, trs = jaccard_tools.calc_jaccard(model, img_val, msk_val)
    else:
        print('loading model')
        model = get_unet()
        model.load_weights(config.glb_base_dir + '/weights/model-17-0.02772.hdf5')
        if generate_jaccard:
            # score, trs = calculate_optimum_binary_thresholds(image_db, model, calc_size=0)
            img_val, msk_val = image_db.get_random_patches(2500, ISZ)
            score, trs = jaccard_tools.calc_jaccard(model, img_val, msk_val)
            print('saving thresholds and score')
            pickle.dump(trs, open(config.glb_base_dir + '/weights/thresh.pickle', 'wb'))
            pickle.dump(score, open(config.glb_base_dir + '/weights/score.pickle', 'wb'))
        else:
            print('loading thresholds and score')
            trs = pickle.load(open(config.glb_base_dir + '/weights/thresh.pickle', 'rb'))
            score = pickle.load(open(config.glb_base_dir + '/weights/score.pickle', 'rb'))
        print('score', score)
        print('thresholds', trs)

    if build_masks_and_submissions:
        mask_dir_name = config.glb_base_dir + '/msk'
        predict_and_write_masks(model, trs, mask_dir_name, IMAGE_SF)
        make_submission_file_from_masks(mask_dir_name, sub_file_name=config.glb_base_dir + '/subm/sub_new_v0.csv')

    # bonus
    check_image_id = '6120_2_2'
    # check_predict(check_image_id, 0, model, trs, IMAGE_SF)
    mask_dir_name = config.glb_base_dir + '/msk'
    check_predict_2(model, trs, check_image_id, 0, mask_dir_name, IMAGE_SF)
