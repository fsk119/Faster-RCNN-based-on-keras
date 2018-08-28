"""
    The idea is from ...
    I use numpy to implement nms and get the regression output of ROI.
    It's important to point out that there is no method like np.where in keras backend.
    So it's difficult to use K.gather function to slice the data we are interested in.
    Maybe we can implement it using tf backend later.
    Because we use numpy to implement, it needs to spend time conveying data between GPU and CPU.
"""
import numpy as np
from RPN_feeder_loss import pred2bbox, IoU_matrix, IoU, bbox2pred, genBaseAnchors
def getROIv1(scores, regression, baseAnchors, numAnchors, imgInfo, stride, threshold):
    """
    use threshold to filter the background boxes
    :param scores:
    :param regression:
    :param baseAnchors:
    :param numAnchors:
    :param imgInfo:
    :param stride:
    :param threshold:
    :return:
    """
    imgBatchIndices, anchorsIndices = np.where(scores > threshold)[:2]
    roiRegression = regression[imgBatchIndices, anchorsIndices, :]
    # indices structure : imgBatch x (H x W x numAnchors)
    featureInfo = [np.floor(dim / stride).astype('int') for dim in imgInfo]
    Y = np.floor(anchorsIndices / (featureInfo[0] * numAnchors))
    X = np.floor((anchorsIndices % (featureInfo[0] * numAnchors)) / numAnchors)
    Z = anchorsIndices % numAnchors
    shifts = np.vstack([X, Y, X, Y]).transpose() * stride
    selectedAnchors = baseAnchors[Z] + shifts
    roi = pred2bbox(roiRegression, selectedAnchors)
    assert roi.ndim == 2
    # modify the boundaries of bboxes
    roi[:, [0, 2]] = np.clip(roi[:, [0, 2]], 0, imgInfo[0])
    roi[:, [1, 3]] = np.clip(roi[:, [1, 3]], 0, imgInfo[1])
    roi = np.hstack([np.expand_dims(imgBatchIndices, axis=-1), roi])
    return roi, scores[imgBatchIndices, anchorsIndices]

def getROI(scores, regression, baseAnchors, numAnchors, imgInfo, stride, topN, nmsFunc=None):
    assert regression.shape[0] == 1
    # only support imageBatch == 1
    N = np.minimum(topN, scores.shape[1])
    indices = scores.ravel().argsort()[::-1]
    roiRegression = regression[0, indices, :]
    """
    be careful
    """
    featureInfo = [np.floor(dim / stride).astype('int') for dim in imgInfo]
    Y = np.floor(indices / (featureInfo[0] * numAnchors))
    X = np.floor((indices % (featureInfo[0] * numAnchors)) / numAnchors)
    Z = indices % numAnchors
    shifts = np.vstack([X, Y, X, Y]).transpose() * stride
    selectedAnchors = baseAnchors[Z] + shifts
    roi = pred2bbox(roiRegression, selectedAnchors)
    roi[:, [0, 2]] = np.clip(roi[:, [0, 2]], 0, imgInfo[0]-1)
    roi[:, [1, 3]] = np.clip(roi[:, [1, 3]], 0, imgInfo[1]-1)
    roi = np.round(roi).astype('int')
    area = np.prod(roi[:, 2:] - roi[:, :2] , axis=1)
    minArea = np.prod(np.array(imgInfo)) * 0
    smallRegionIndices = np.where(np.all([area>=minArea, area>0, np.all(roi[:, 2:]-roi[:, :2]>stride, axis=1)],
                                         axis=0))[0][:N]
    indices = indices[smallRegionIndices]
    roi = roi[smallRegionIndices, :]
    selectedAnchors = selectedAnchors[smallRegionIndices, :]
    if nmsFunc:
        nmsIndices = nmsFunc(roi)
        indices = indices[nmsIndices]
        roi = roi[nmsIndices, :]
        selectedAnchors = selectedAnchors[nmsIndices, :]
    return roi, indices, selectedAnchors

def nmsWrapper(threshold):
    def nms(roi):
        N = roi.shape[0]
        isValid = np.ones([N, 1])
        roiIndices = []
        for i in range(N):
            if not isValid[i]:
                continue
            roiIndices.append(i)
            indices = np.arange(i+1, N)
            overlaps = IoU_matrix(np.repeat(np.expand_dims(roi[i, :], axis=0), indices.shape[0], axis=0),
                           roi[indices, :])
            indices = np.where(overlaps >= threshold)[0]+(i+1)
            isValid[indices] = 0
        return roiIndices
    return nms


def recallMetricvNMS(y_true, y_pred):
    raise  NotImplementedError()

def getROILabel(roi, gtBBoxes, gtLabels, forMetric=False):
    # same regulations as ...
    N, M = roi.shape[0], gtBBoxes.shape[0]
    overlaps = np.zeros([N, M])
    for j in range(M):
        overlaps[:, j] = IoU_matrix(roi, np.expand_dims(gtBBoxes[j, :], axis=0))
    if forMetric:
        return np.mean(np.max(overlaps, -1))
    labels = np.empty([N, 1])
    labels.fill(-1)
    BBoxes = np.zeros([N, 4])

    # 1) >=0.7 positive
    rowMaxInd = np.argmax(overlaps, axis=1)
    rowMaxValue = overlaps[np.arange(N), rowMaxInd]
    indices = rowMaxValue>=0.6
    labels[indices] = gtLabels[rowMaxInd[indices]]
    BBoxes[indices] = gtBBoxes[rowMaxInd[indices], :]
    # negative bboxes <=0.3
    indices = rowMaxValue<=0.3
    labels[indices] = 0
    # 2) every bbox have one pred bbox at least
    colMaxInd = np.argmax(overlaps, axis=0)
    labels[colMaxInd] = gtLabels
    BBoxes[colMaxInd] = gtBBoxes
    indices = np.where(labels>=0)[0]
    selectedROI = roi[indices]
    selectedLabels = labels[indices]
    # construct bbox regression  matrix
    # based on roi ?
    # not correct
    indices = np.where(labels>0)[0]
    BBoxes[indices,] = bbox2pred(roi[indices, :], BBoxes[indices, :])
    # _unmap([selectedROI.shape[0], 4], indices, selectedTargetBBoxes, 0)
    selectedTargetBBoxes = np.concatenate([selectedLabels, BBoxes[np.squeeze(labels>=0, axis=1), :]],
                                          axis=-1)

    return selectedROI, selectedLabels, selectedTargetBBoxes

def getROILabelv2(roi, gtBBoxes, gtLabels, forMetric=False):
    N, M = roi.shape[0], gtBBoxes.shape[0]
    overlaps = np.zeros([N, M])
    for j in range(M):
        overlaps[:, j] = IoU_matrix(roi, np.expand_dims(gtBBoxes[j, :], axis=0))
    if forMetric:
        return np.mean(np.max(overlaps, -1))
    x_ROI = []
    targetBBoxes = []
    targetLabels = []
    bestOverlap = np.max(overlaps, axis=1)
    maxOverlapIndices = np.argmax(overlaps, axis=1)
    cntGtBBoxes = np.zeros([M, 1])
    for i in range(N):
        if bestOverlap[i]>=0.55:
            x_ROI.append(roi[i, :])
            targetLabels.append(gtLabels[maxOverlapIndices[i]])
            targetBBoxes.append(np.concatenate(
                [targetLabels[-1], gtBBoxes[maxOverlapIndices[i], :]], axis=-1))
            cntGtBBoxes[maxOverlapIndices[i]] += 1
        elif bestOverlap[i]<0.35:
            continue
        else:
            x_ROI.append(roi[i, :])
            targetLabels.append(0)
            targetBBoxes.append(np.zeros([5, ]))

    for j in range(M):
        if cntGtBBoxes[j]>0:
            continue
        indices = np.argmax(overlaps[:, j])
        x_ROI.append(roi[indices, :])
        targetLabels.append(gtLabels[j])
        targetBBoxes.append(np.concatenate(
            [targetLabels[-1], gtBBoxes[j, :]], axis=-1))
    # compute targetBBoxes
    x_ROI = np.array(x_ROI)
    targetLabels = np.reshape(np.array(targetLabels), [-1, 1])
    targetBBoxes = np.array(targetBBoxes)
    indices = np.where(targetLabels>0)[0]
    targetBBoxes[indices, 1:] = bbox2pred(x_ROI[indices], targetBBoxes[indices, 1:] )
    return x_ROI, targetLabels, targetBBoxes

class ROILayer:
    ## caution! This model only supports batchsize == 1
    def __init__(self, mode, nms, stride=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
        self.mode = mode
        self.nms = nms
        self.stride = stride
        self.baseAnchors = genBaseAnchors(stride, ratios, scales)

    def __call__(self, scores, regression, imgInfo, topN, **kwargs):
        numAnchors = self.baseAnchors.shape[0]
        roi, indices, selectedAnchors = \
            getROI(scores, regression, self.baseAnchors, numAnchors,
                   imgInfo, self.stride, topN, self.nms)
        if self.mode == 'train':
            # still need to get labels for every roi
            gtBBoxes = kwargs.get('gtBBoxes')
            gtLabels = kwargs.get('gtLabels')
            forMetric = kwargs.get('forMetric', False)
            if gtBBoxes is None or gtLabels is None:
                raise ValueError('gtbboxes or gtlabels is None')
            isSample = kwargs.get('isSample', False)
            if forMetric:
                return getROILabelv2(roi, gtBBoxes, gtLabels, forMetric=forMetric)
            roi, labels, targetBBoxes =\
                getROILabelv2(roi, gtBBoxes, gtLabels)
            if isSample:
                # delete some bg classes
                N = roi.shape[0]
                indices = np.where(labels==0)[0]
                if indices.shape[0]*2>N:
                    ignoreIndices = np.random.choice(indices, 2*indices.shape[0]-N)
                    roi = np.delete(roi, ignoreIndices, axis=0)
                    labels = np.delete(labels, ignoreIndices, axis=0)
                    targetBBoxes = np.delete(targetBBoxes, ignoreIndices, axis=0)
            return roi, labels, targetBBoxes
        return roi, indices, selectedAnchors
"""
[[384 309 470 477]
 [  6 389 106 597]
 [419 336 517 541]
 [  6 389 106 597]
 [384 309 470 477]]
"""

if __name__=='__main__':
    import pickle
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    layer = ROILayer('train', nmsWrapper(0.7), )
    #with open('testROI.pickle', 'rb') as f:
    #    data = pickle.load(f)
    roi, labels, targetBBoxes = layer(data['predScores'], data['regressionAnchors'], [800, 600],
          20, gtBBoxes=data['gtBBoxes'][0, :, :4],
          gtLabels=np.expand_dims(data['gtBBoxes'][0, :, -1], axis=-1), isSample=True)
    #roi, indices, targetBBoxes = layer(data[0], data[1], [800, 600], 100)
    print(labels, targetBBoxes)
