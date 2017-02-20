import cv2
import os
import tifffile as tiff
import numpy as np
from config import glb_base_dir
from skimage.transform import rescale
import skimage.color as color


def image_resize(img_mat, sf=1.0):
    width = int(round(img_mat.shape[1] * sf))
    height = int(round(img_mat.shape[0] * sf))
    img2 = cv2.resize(img_mat, (width, height), interpolation=cv2.INTER_CUBIC)
    return img2


def get_image_from_id(image_id, img_resize):
    # img_mat = get_image_3_from_id(image_id, img_resize)
    img_mat = get_image_pansharpen_from_id(image_id, img_resize)
    # img_mat = get_image_20d_from_id(image_id, img_resize)
    # img_mat = get_image_4d_from_id(image_id, img_resize)
    return img_mat


def get_image_m_from_id(image_id):
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(glb_base_dir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img_mat = tiff.imread(filename)
    img_mat = np.rollaxis(img_mat, 0, 3)
    return img_mat


def get_image_3_from_id(image_id, image_sf):
    filename = os.path.join(glb_base_dir, 'three_band', '{}.tif'.format(image_id))
    img_mat = tiff.imread(filename)  # shape = (3,33xx,33yy)
    img_mat = np.rollaxis(img_mat, 0, 3)
    img_mat = image_resize(img_mat, image_sf)
    return img_mat


def get_image_pansharpen_from_id(image_id, image_sf):
    filename = os.path.join(glb_base_dir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img_m = tiff.imread(filename)  # shape = (8,837,851)

    filename = os.path.join(glb_base_dir, 'sixteen_band', '{}_P.tif'.format(image_id))
    img_p = tiff.imread(filename)  # shape = (3,33xx,33yy)

    img_x = pansharpen(img_m, img_p)

    img_mat = image_resize(img_x, image_sf)
    return img_mat


def get_image_20d_from_id(image_id, m_resize):
    filename = os.path.join(glb_base_dir, 'three_band', '{}.tif'.format(image_id))
    img_3b = tiff.imread(filename)  # shape = (3,33xx,33yy)
    img_3b_roll = np.rollaxis(img_3b, 0, 3)
    img_3b_resize = cv2.resize(img_3b_roll, (m_resize, m_resize), interpolation=cv2.INTER_CUBIC)

    filename = os.path.join(glb_base_dir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img_m = tiff.imread(filename)  # shape = (3,33xx,33yy)
    img_m_roll = np.rollaxis(img_m, 0, 3)
    img_m_resize = cv2.resize(img_m_roll, (m_resize, m_resize), interpolation=cv2.INTER_CUBIC)

    filename = os.path.join(glb_base_dir, 'sixteen_band', '{}_A.tif'.format(image_id))
    img_a = tiff.imread(filename)  # shape = (3,33xx,33yy)
    img_a_roll = np.rollaxis(img_a, 0, 3)
    img_a_resize = cv2.resize(img_a_roll, (m_resize, m_resize), interpolation=cv2.INTER_CUBIC)

    filename = os.path.join(glb_base_dir, 'sixteen_band', '{}_P.tif'.format(image_id))
    img_p = tiff.imread(filename)  # shape = (3,33xx,33yy)
    img_p_resize = cv2.resize(img_p, (m_resize, m_resize), interpolation=cv2.INTER_CUBIC)

    img_mat = np.zeros((m_resize, m_resize, 20), 'uint8')
    img_mat[:, :, 0:3] = img_3b_resize
    img_mat[:, :, 3] = img_p_resize
    img_mat[:, :, 4:12] = img_m_resize
    img_mat[:, :, 12:21] = img_a_resize

    return img_mat


def get_image_4d_from_id(image_id, m_resize):
    filename = os.path.join(glb_base_dir, 'three_band', '{}.tif'.format(image_id))
    img_3b = tiff.imread(filename)  # shape = (3,33xx,33yy)
    img_3b_roll = np.rollaxis(img_3b, 0, 3)
    img_3b_resize = cv2.resize(img_3b_roll, (m_resize, m_resize), interpolation=cv2.INTER_CUBIC)

    filename = os.path.join(glb_base_dir, 'sixteen_band', '{}_P.tif'.format(image_id))
    img_p = tiff.imread(filename)  # shape = (3,33xx,33yy)
    img_p_resize = cv2.resize(img_p, (m_resize, m_resize), interpolation=cv2.INTER_CUBIC)

    img_mat = np.zeros((m_resize, m_resize, 4), 'uint8')
    img_mat[:, :, 0:3] = img_3b_resize
    img_mat[:, :, 3] = img_p_resize

    return img_mat


def get_image_am_from_id(image_id):
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(glb_base_dir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img_m = tiff.imread(filename)
    img_m = np.rollaxis(img_m, 0, 3)

    filename = os.path.join(glb_base_dir, 'sixteen_band', '{}_A.tif'.format(image_id))
    img_a = tiff.imread(filename)
    img_a = np.rollaxis(img_a, 0, 3)

    img_mat = np.append(img_a, img_m, axis=0)
    return img_mat


def stretch_n(bands, lower_percent=5, higher_percent=95):
    out = np.zeros_like(bands).astype(np.float32)  # fms from post
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)


# functions
def stretch(bands, lower_percent=2, higher_percent=98):
    out = np.zeros_like(bands)
    for i in range(3):
        a = 0
        b = 255
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.uint8)


def pansharpen(m, pan, method='browley', W=0.1, all_data=False):
    # get m_bands
    rgbn = np.empty((m.shape[1], m.shape[2], 4))
    rgbn[:, :, 0] = m[4, :, :]  # red
    rgbn[:, :, 1] = m[2, :, :]  # green
    rgbn[:, :, 2] = m[1, :, :]  # blue
    rgbn[:, :, 3] = m[6, :, :]  # NIR-1

    # scaled them
    rgbn_scaled = np.empty((m.shape[1] * 4, m.shape[2] * 4, 4))

    for i in range(4):
        img = rgbn[:, :, i]
        scaled = rescale(img, (4, 4))
        rgbn_scaled[:, :, i] = scaled

    # check size and crop for pan band
    if pan.shape[0] < rgbn_scaled.shape[0]:
        rgbn_scaled = rgbn_scaled[:pan.shape[0], :, :]
    else:
        pan = pan[:rgbn_scaled.shape[0], :]

    if pan.shape[1] < rgbn_scaled.shape[1]:
        rgbn_scaled = rgbn_scaled[:, :pan.shape[1], :]
    else:
        pan = pan[:, :rgbn_scaled.shape[1]]

    R = rgbn_scaled[:, :, 0]
    G = rgbn_scaled[:, :, 1]
    B = rgbn_scaled[:, :, 2]
    I = rgbn_scaled[:, :, 3]

    image = None

    if method == 'simple_browley':
        all_in = R + G + B
        prod = np.multiply(all_in, pan)

        r = np.multiply(R, pan / all_in)[:, :, np.newaxis]
        g = np.multiply(G, pan / all_in)[:, :, np.newaxis]
        b = np.multiply(B, pan / all_in)[:, :, np.newaxis]

        image = np.concatenate([r, g, b], axis=2)

    if method == 'sample_mean':
        r = 0.5 * (R + pan)[:, :, np.newaxis]
        g = 0.5 * (G + pan)[:, :, np.newaxis]
        b = 0.5 * (B + pan)[:, :, np.newaxis]

        image = np.concatenate([r, g, b], axis=2)

    if method == 'esri':
        ADJ = pan - rgbn_scaled.mean(axis=2)
        r = (R + ADJ)[:, :, np.newaxis]
        g = (G + ADJ)[:, :, np.newaxis]
        b = (B + ADJ)[:, :, np.newaxis]
        i = (I + ADJ)[:, :, np.newaxis]

        image = np.concatenate([r, g, b, i], axis=2)

    if method == 'browley':
        DNF = (pan - W * I) / (W * R + W * G + W * B)

        r = (R * DNF)[:, :, np.newaxis]
        g = (G * DNF)[:, :, np.newaxis]
        b = (B * DNF)[:, :, np.newaxis]
        i = (I * DNF)[:, :, np.newaxis]

        image = np.concatenate([r, g, b, i], axis=2)

    if method == 'hsv':
        hsv = color.rgb2hsv(rgbn_scaled[:, :, :3])
        hsv[:, :, 2] = pan - I * W
        image = color.hsv2rgb(hsv)

    if all_data:
        return rgbn_scaled, image, I
    else:
        return image
