import numpy as np
import keras.backend as K
def pred2bbox(deltas, boxes):
    """
    # numpy version
    wa = anchors[:, 2]-anchors[:, 0]+1
    ha = anchors[:, 3]-anchors[:, 1]+1
    w = np.exp(pred[:, 2])*wa
    h = np.exp(pred[:, 3])*ha
    x = pred[:, 0]*wa+anchors[:, 0]
    y = pred[:, 1]*ha+anchors[:, 1]
    return np.vstack([x, y, x+w-1, h+y-1]).transpose()
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def bbox2pred(ex_rois, gt_rois):
    """
    wa = anchors[:, 2]-anchors[:, 0]+1
    ha = anchors[:, 3]-anchors[:, 1]+1
    w_ = bbox[:, 2]-bbox[:, 0]+1
    h_ = bbox[:, 3]-bbox[:, 1]+1
    tx = (bbox[:, 0]-anchors[:, 0])/wa
    ty = (bbox[:, 1]-anchors[:, 1])/ha
    tw = np.log(w_/wa)
    th = np.log(h_/ha)
    return np.vstack([tx, ty, tw, th]).transpose()
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def anchors2HWXY(anchors):
    H = anchors[2]-anchors[0]+1
    W = anchors[3]-anchors[1]+1

    X = anchors[0]+0.5*(H-1)
    Y = anchors[1]+0.5*(W-1)
    return np.hstack([H, W, X, Y])

def HWXY2anchors(HWXY):
    x1 = HWXY[2]-0.5*(HWXY[0]-1)
    y1 = HWXY[3]-0.5*(HWXY[1]-1)
    x2 = HWXY[2]+0.5*(HWXY[0]-1)
    y2 = HWXY[3]+0.5*(HWXY[1]-1)
    return np.hstack([x1, y1, x2, y2])

def sampler(labels, batchsize=32, ratio=0.5):
    positiveNum = int(batchsize*ratio)
    positiveIndices = np.where(labels==1)[0]
    cnt = positiveIndices.shape[0]
    if cnt>positiveNum:
        ind = np.random.choice(positiveIndices, cnt-positiveNum, False)
        labels[ind] = -1
        cnt = positiveNum
    zeroIndices = np.where(labels==0)[0]
    if zeroIndices.shape[0]>batchsize-cnt:
        ind = np.random.choice(zeroIndices, zeroIndices.shape[0]-(batchsize-cnt), False)
        labels[ind] = -1
    return labels

def genBaseAnchors(basesize, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
    """
    in ratio and scale adjust, we keep the center of the anchor unchanged

    :param basesize:
    :param ratios:
    :param scales:
    :return:
    """

    baseAnchors = np.array([0, 0, basesize-1, basesize-1])

    # wrap ratios and scales
    ratios = np.array(ratios)
    scales = np.array(scales)
    _anchors_ratio = []
    hwxy = anchors2HWXY(baseAnchors)
    for ratio in ratios:
        # keep the size unchanged
        s = hwxy[0]*hwxy[1]
        HWXY = hwxy.copy()
        HWXY[0] = np.round(np.sqrt(s/ratio), )
        HWXY[1] = np.round(HWXY[0]*ratio, )
        _anchors_ratio.append(HWXY2anchors(HWXY))
    _anchors = []
    for anchor in _anchors_ratio:
        hwxy = anchors2HWXY(anchor)
        for scale in scales:
            HWXY = hwxy.copy()
            HWXY[0] *= scale
            HWXY[1] *= scale
            _anchors.append(HWXY2anchors(HWXY))
    return np.array(_anchors).astype(np.int)

def IoU(y_pred, y_true, smooth=1):
    """
    :param y_pred: [x_upperleft, y_upperleft, x_downright, y_downright]
    :param y_true: [x_upperleft, y_upperleft, x_downright, y_downright]
    :return: intersection over union
    """
    intersection_upperleft = np.max([y_pred[0:2], y_true[0:2]], axis=0)
    intersection_downright = np.min([y_pred[2:], y_true[2:]], axis=0)

    if np.any(intersection_upperleft>=intersection_downright):
        intersection = 0
    else:
        intersection = np.prod(intersection_downright-intersection_upperleft+smooth)
    areaBoxA = np.prod(y_pred[2:]-y_pred[0:2]+smooth)
    areaBoxB = np.prod(y_true[2:]-y_true[0:2]+smooth)
    union = areaBoxA+areaBoxB-intersection
    return  (intersection)/(union)

def IoUfast(y_pred, y_true, smooth=1):
    """

    :param y_pred: dim==2
    :param y_true: dim=2
    :param smooth:
    :return:
    """
    intersection_upperleft = np.maximum(y)
    pass

def IoU_matrix(y_pred, y_true, smooth=1):
    """

    :param y_pred: N x 4
    :param y_true: N x 4
    :param smooth: constant 1
    :return: IoU N x 1
    """
    areaA = np.prod(y_pred[:, :2]-y_pred[:, 2:]+smooth, axis=1)
    areaB = np.prod(y_true[:, :2]-y_true[:, 2:]+smooth, axis=1)
    upperLeft = np.maximum(y_pred[:, :2], y_true[:, :2])
    downRight = np.minimum(y_pred[:, 2:], y_true[:, 2:])
    indices = np.where(np.any(upperLeft>downRight, axis=1))[0]
    intersection = np.prod((downRight-upperLeft+smooth), axis=1)
    intersection[indices] = 0
    return intersection/(areaA+areaB-intersection)

def _unmap(originSize, indices, value, fill=-1):
    origin = np.empty(originSize)
    origin.fill(fill)
    origin[indices, ] = value
    return origin

def genAnchorLabel(imgInfo, stride, gtboxes, baseAnchors, isSample=False, batchsize=32, ratio=0.4, debug=True):
    # imgInfo is of feature map
    featureDims = [np.floor(dim/stride).astype('int') for dim in imgInfo]
    K = baseAnchors.shape[0]
    A = featureDims[0]*featureDims[1]
    N = gtboxes.shape[0]
    # gen anchors using meshgrid
    x = np.arange(0, featureDims[0])*stride
    y = np.arange(0, featureDims[1])*stride

    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten(), Y.flatten()

    coordinates = np.vstack([X, Y, X, Y]).transpose()

    shiftedAnchors = np.reshape(coordinates, [A, 1, 4]) + \
        np.reshape(baseAnchors, [1, K, 4])
    allAnchors = np.reshape(shiftedAnchors, [K*A, 4])
    insideInd = np.where((allAnchors[:, 0]>=0) & (allAnchors[:, 1]>=0) &
                         (allAnchors[:, 2]<imgInfo[0]) &
                         (allAnchors[:, 3]<imgInfo[1]))[0]
    insideNum = len(insideInd)
    overlaps = np.zeros([insideNum, N])
    #for i in range(insideNum):
        #for j in range(N):
        #    overlaps[i, j] = IoU(allAnchors[insideInd[i],], gtboxes[j,])
    for j in range(N):
        overlaps[:, j] = IoU_matrix(allAnchors[insideInd, :], np.expand_dims(gtboxes[j, ], axis=0))

    labels = np.empty([insideNum, ])
    labels.fill(-1)
    # positive label
    rowMaxInd = np.argmax(overlaps, axis=1)
    rowMaxVal = overlaps[np.arange(overlaps.shape[0]), rowMaxInd]
    rowPositiveInd = rowMaxVal>=0.5 # principle 1
    labels[rowPositiveInd] = 1
    # negative label
    rowNegativeInd = overlaps[np.arange(overlaps.shape[0]), rowMaxInd]<0.3
    labels[rowNegativeInd] = 0

    gtBBoxInd = np.empty_like(labels)
    gtBBoxInd.fill(-1)
    gtBBoxInd[rowPositiveInd] = rowMaxInd[rowPositiveInd]
    nonAnchorsBoxes = np.array(list(set(np.arange(N))-set(rowMaxInd[rowPositiveInd])))
    if nonAnchorsBoxes.shape[0]:
        colMaxInd = np.argmax(overlaps[:, nonAnchorsBoxes], axis=0)
        labels[colMaxInd] = 1  # principle 2
        gtBBoxInd[colMaxInd] = nonAnchorsBoxes

    if isSample:
        labels = sampler(labels, batchsize, ratio)
    gtBBoxInd = gtBBoxInd[labels > 0].astype('int')
    # for postive label we need to construct their gt bbox matrix
    # shape of labels is [(H x W x C), ]
    positiveInd = np.where(labels>0)[0]
    # positiveInd value == gtBBoxInd value [(i th location)] -->
    gtBBox = gtboxes[gtBBoxInd, :]
    targetBBox = bbox2pred(allAnchors[insideInd[positiveInd], :], gtBBox)
    positiveAnchors = allAnchors[insideInd[positiveInd], :]

    labels = _unmap([K*A, ], insideInd, labels, -1)
    targetBBox = _unmap([A*K, 4], insideInd[positiveInd], targetBBox, 0)
    labelsValid = (labels>=0)
    bboxesValid = np.expand_dims(labels>0, axis=1)
    # print(targetBBox.shape, bboxesValid.shape, )
    # print(labels.shape, labelsValid.shape)
    labels = np.stack([labelsValid, labels], axis=1)
    targetBBox = np.concatenate([bboxesValid, targetBBox], axis=1)
    # print(labels.shape, targetBBox.shape)
    # labels = np.reshape(labels, [*imgInfo, K,])

    return labels, targetBBox
           #insideInd[positiveInd], insideInd[negativeInd]

def smooth_l1_loss(y_true, y_pred):
    sigma = 3.0
    sigma2 = sigma*sigma
    isValid = K.expand_dims(y_true[:, :, 0], 2)
    isValid = K.repeat_elements(isValid, 4, -1)
    # flatten data
    # y_pred = K.reshape(y_pred, [-1, 4])
    d = K.abs(y_pred - y_true[:, :, 1:])
    isLess = K.cast(d<1/sigma2, 'float32')
    left = 0.5*isLess*K.square(sigma*d)
    right = (1-isLess)*(d-0.5/sigma2)
    res = K.sum((left+right)*isValid)/K.sum(isValid)
    return res

def crossEntropy(y_true, y_pred):
    # y_pred = K.flatten(y_pred)
    entropy = K.binary_crossentropy(y_true[:, :, 1], K.squeeze(y_pred, axis=-1))
    return K.sum(y_true[:, :, 0]*entropy)/K.sum(y_true[:, :, 0])

def wCEWrapper(positiveWeight):
    def weigthedCE(y_true, y_pred):
        entropy = K.binary_crossentropy(y_true[:, :, 1],K.squeeze(y_pred, axis=-1))
        wVector = y_true[:, :, 1]*positiveWeight+(1-y_true[:, :, 1])*1
        # positive weight is  , zero weight is 1
        return K.sum(y_true[:, :, 0]*entropy*wVector)/K.sum(y_true[:, :, 0])
    return weigthedCE

def f1Metric(y_true, y_pred):
    # metric for score
    # y_pred: batch x (H x W x numAnchors) x 1
    # y_true: batch x (H x W x numAnchors) x 2
    isValid = K.expand_dims(y_true[:, :, 0], axis=-1)
    pred = K.cast(y_pred > 0.5, 'float32')
    y_true = K.expand_dims(y_true[:, :, -1], axis=-1)
    TP = K.sum(y_true * pred * isValid)
    FP = K.sum((1 - y_true) * pred * isValid)
    FN = K.sum(y_true * (1 - pred) * isValid)
    precision = (TP + 1) / (FP + TP + 1)
    recall = (TP + 1) / (TP + FN + 1)
    f1 = 1 / (1 / precision + 1 / recall + 1)
    return f1

def recallMetric(y_true, y_pred):
    isValid = K.expand_dims(y_true[:, :, 0], axis=-1)
    pred = K.cast(y_pred > 0.5, 'float32')
    y_true = K.expand_dims(y_true[:, :, 1], axis=-1)
    TP = K.sum(y_true * pred * isValid)
    FN = K.sum(y_true * (1 - pred) * isValid)
    recall = (TP + 1) / (TP + FN + 1)
    return recall



