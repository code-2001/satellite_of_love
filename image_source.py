import cv2
import os
import tifffile as tiff
import numpy as np
from config import glb_base_dir


def image_resize(img_mat, sf=1.0):
    width = int(round(img_mat.shape[1] * sf))
    height = int(round(img_mat.shape[0] * sf))
    img2 = cv2.resize(img_mat, (width, height), interpolation=cv2.INTER_CUBIC)
    return img2


def get_image_from_id(image_id, img_resize):
    img_mat = get_image_3_from_id(image_id, img_resize)
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
