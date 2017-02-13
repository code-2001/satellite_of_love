from sklearn.metrics import jaccard_similarity_score
from keras import backend as kb
from config import smooth


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = kb.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = kb.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return kb.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = kb.round(kb.clip(y_pred, 0, 1))
    intersection = kb.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = kb.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return kb.mean(jac)


def calc_jaccard(jacc_model, img_mat, msk_mat, num_class=10):
    print('-- calculating jaccard on', img_mat.shape[0], 'samples of size', img_mat.shape[2], 'x', img_mat.shape[3])

    prd = jacc_model.predict(img_mat, batch_size=4)
    avg_vec, thresh_vec = [], []

    for i in range(num_class):
        t_msk = msk_mat[:, i, :, :]  # get mask for class
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.reshape(msk_mat.shape[0] * msk_mat.shape[2], msk_mat.shape[3])  # reshape to remove dim 1
        t_prd = t_prd.reshape(msk_mat.shape[0] * msk_mat.shape[2], msk_mat.shape[3])
        # find the best threshold using a linear search
        best_jacc, best_thresh = 0, 0
        for j in range(10):
            thresh = j / 10.0  # threshold
            pred_binary_mask = t_prd > thresh

            jacc = jaccard_similarity_score(t_msk, pred_binary_mask)
            if jacc > best_jacc:
                best_jacc = jacc  # best jaccard value
                best_thresh = thresh  # best threshold
        print('  cls', i, 'jacc', best_jacc, 'thresh', best_thresh)
        avg_vec.append(best_jacc)
        thresh_vec.append(best_thresh)

    jacc_score = sum(avg_vec) / num_class
    print('-- average jaccard score', jacc_score)
    return jacc_score, thresh_vec
