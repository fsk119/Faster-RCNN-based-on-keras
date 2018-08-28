from clsNetwork import RPNPooling, clsCEWrapper, clsSLWrapper, mAPMetricWrapper, clsWeigtedCEWrapper
from vocDataGen import Generator
from nms import ROILayer, nmsWrapper, getROILabel
from RPN_feeder_loss import smooth_l1_loss, wCEWrapper, recallMetric
from buildRPN import VGG16, RPN
from keras import Model, regularizers
from keras.layers import Flatten, Dense, Reshape, Dropout, Input
from keras import optimizers
import numpy as np
import os
import pickle
import keras.backend as K
from RPN_feeder_loss import pred2bbox
# this file is used to train the whole model
def buildClsNetwork(baseLayer, roi, nb_classes, poolsize=7, trainable=False):
    rpnpoolLyaer = RPNPooling(poolsize)
    out = rpnpoolLyaer([baseLayer, roi])
    out._keras_shape = K.int_shape(out) # caution
    flattenLayer = Flatten()
    out = flattenLayer(out)
    out = Dense(4096, activation='relu', name='fc1',)(out)
    out = Dropout(0.5)(out)
    out = Dense(4096, activation='relu', name='fc2',)(out)
    out = Dropout(0.5)(out)

    scores = Dense(nb_classes, activation='softmax', kernel_initializer='zero',
                   name='clsScore')(out)
    regression = Dense(4*nb_classes,  activation='linear', name='clsRegression',
                       kernel_initializer='zero',)(out)

    return scores, regression

def buildModel(imgInput, roiInput, nb_classes, transfer):
    encoder = VGG16(imgInput, not transfer)
    scores, regression = RPN(encoder, 9)
    scores = Reshape([-1, 1], name='RPNScores')(scores)
    regression = Reshape([-1, 4], name='RPNRegression')(regression)

    clsScores, clsRegression = buildClsNetwork(encoder, roiInput, nb_classes, trainable=not transfer)
    rpn = Model(inputs=imgInput, outputs=[scores, regression])
    clsNet = Model(inputs=[imgInput, roiInput], outputs=[clsScores, clsRegression])
    allModel = Model(inputs=[imgInput, roiInput], outputs=[scores, regression, clsScores, regression])
    return rpn, clsNet, allModel

class FRCNN:
    def __init__(self, nb_classes, loadPath, transfer, mode='test'):
        img_input = Input(shape=[None, None, 3])
        roi_input = Input(shape=[None, 4])
        models = buildModel(img_input, roi_input, nb_classes, transfer)
        self.rpn = models[0]
        self.clsNet = models[1]
        self.allModel = models[2]
        self.models = models # for convenience
        self.nb_classes = nb_classes
        if loadPath:
            self.allModel.load_weights(loadPath, by_name=True)

        self.nms = nmsWrapper(0.5)
        self.roilayer = ROILayer(mode, None)

    def predict(self, image, imgInfo=[800, 600], topN=200, threshold=0.5):
        if image.ndim==3:
            image = np.expand_dims(image, axis=0)
        predScores, predRegression = self.rpn.predict_on_batch(image)
        self.roilayer.mode = 'test'
        roi, _, _ = \
            self.roilayer(predScores, predRegression, imgInfo, topN)
        batchROI = np.round(np.expand_dims(roi, axis=0) / 16)

        finalScores, finalRegression = self.clsNet.predict_on_batch([image, batchROI])
        finalScores, finalRegression = np.squeeze(finalScores), np.reshape(np.squeeze(finalRegression),
                                                                           [-1, self.nb_classes, 4])
        """
        labels = np.argmax(finalScores, axis=-1)
        indices = np.where(labels>0)[0]
        _indices = np.argsort(finalScores[indices, labels[indices]])[::-1]
        indices = indices[_indices]
        selectedROI = roi[indices]
        selectedLabels = labels[indices]
        selectedRegression = np.reshape(finalRegression, [-1, self.nb_classes, 4])[indices, selectedLabels, :]
        positiveBBoxes = pred2bbox(selectedRegression, selectedROI)
        nmsIndices = self.nms(positiveBBoxes)
        finalBBoxes = positiveBBoxes[nmsIndices, :]
        finalLabels = selectedLabels[nmsIndices]
        """
        selectedLabel, selectedBBoxes = np.empty([0, ]), np.empty([0, 4])
        for clsInd in range(1, self.nb_classes):
            clsScores = finalScores[:, clsInd]
            indices = np.where(clsScores >= threshold)[0]
            print(indices.shape, finalRegression.shape)
            realBBoxes = pred2bbox(finalRegression[indices, clsInd, :], roi[indices, :])
            realBBoxes = realBBoxes[np.argsort(clsScores[indices])[::-1]]
            nmsIndices = self.nms(realBBoxes)
            if (len(nmsIndices) == 0):
                continue
            print(selectedLabel.shape, np.array([clsInd] * len(nmsIndices)).shape)
            selectedLabel = np.concatenate([selectedLabel, np.array([clsInd] * len(nmsIndices))])
            selectedBBoxes = np.vstack([selectedBBoxes, realBBoxes[nmsIndices]])
        return np.array(selectedLabel), np.array(selectedBBoxes)

    def testRPN(self, image, imgInfo, topN):
        """

        """
        if image.ndim==3:
            image = np.expand_dims(image, axis=0)
        predScores, predRegression = self.rpn.predict_on_batch(image)
        self.roilayer.mode = 'train'
        roi, _, _ = \
            self.roilayer(predScores, predRegression, imgInfo, topN)
        return roi
# hope everything is ok
def train(models, gen, nb_classes, **kwargs):
    """
    1) model.compile
    2) model train three steps: rpn train, cls train
    :param model:
    :param gen:
    :param kwargs:
    :return:
    """
    assert len(models)==3
    rpn, cls, allModel = models
    epoch = kwargs.get('epoch')
    initEpoch = kwargs.get('init_epoch')
    nmsThreshold = kwargs.get('nmsThreshold', 0.5)
    topN = kwargs.get('topN', 256)
    save_path = kwargs.get('save_path', )
    printStyle = kwargs.get('printStyle', '{0}: {1} ')
    saveStyle = kwargs.get('saveStyle')
    load_path = kwargs.get('loadPath', None)
    freezeRPN = kwargs.get('freezeRPN', False)
    OnlyCls = kwargs.get('OnlyCls', False)
    lr = kwargs.get('lr', 1e-5)
    nms = nmsWrapper(nmsThreshold)
    N = len(gen)
    if freezeRPN:
        for layer in rpn.layers:
            layer.trainable = False

    optimizer = optimizers.Adam(lr)
    #weights = np.ones([21, ])
    #weights[0] = 0.6
    #weights[2] = 0.7
    rpn.compile(optimizer, [wCEWrapper(2), smooth_l1_loss],loss_weights=[1.0, 4.0],
                metrics={'RPNScores': recallMetric})
    cls.compile(optimizer, [clsCEWrapper(nb_classes), clsSLWrapper(nb_classes)],
                loss_weights=[1.0, 1.0],
                )
    #clsWeigtedCEWrapper(nb_classes, weights)
    roilayer = ROILayer('train', nms, )
    N1 = len(rpn.metrics_names)+1
    N2 = len(cls.metrics_names)+2
    rpnMetricNames = rpn.metrics_names+['mIoU']
    clsMetricNames = cls.metrics_names+['mIoU', 'precision']
    if not load_path is None:
        if type(load_path) is list or type(load_path) is tuple:
            assert len(load_path)==2
            cls.load_weights(load_path[1], by_name=True)
            rpn.load_weights(load_path[0], by_name=True)
        else:
            allModel.load_weights(load_path, by_name=True)

    for e in range(initEpoch, epoch):
        rpnLoss = np.zeros([N1,])
        clsLoss = np.zeros([N2,])
        for j in range(N):
            image, labels, targetBBoxes, gtBBoxes = next(gen)
            losses1 = rpn.train_on_batch(image, [labels, targetBBoxes])

            predScores, predRegression = rpn.predict_on_batch(image)
            losses1.append(roilayer(predScores, predRegression, gen.dims,
                        topN, gtBBoxes=gtBBoxes[0, :, :4],
                        gtLabels=np.expand_dims(gtBBoxes[0, :, -1], axis=-1),
                        forMetric = True))
            rpnLoss += np.array(losses1)

            roi, labels, targetBBoxes = \
                roilayer(predScores, predRegression, gen.dims, topN,
                         gtBBoxes = gtBBoxes[0, :, :4],
                         gtLabels = np.expand_dims(gtBBoxes[0, :, -1], axis=-1),
                         isSample = True)
            if OnlyCls:
                roi = pred2bbox(targetBBoxes[:, 1:], roi)
            batchROI = np.round(np.expand_dims(roi, axis=0)/16)
            # labels = np.expand_dims(labels, axis=0)
            # targetBBoxes = np.expand_dims(targetBBoxes, axis=0)
            losses2 = cls.train_on_batch([image, batchROI], [labels, targetBBoxes])
            clsScores, clsRegression = cls.predict_on_batch([image, batchROI])
            clsRegression = np.reshape(clsRegression, [-1, nb_classes, 4])
            indices = targetBBoxes[:, 0].astype('int')
            nonBG = np.where(indices>0)[0]
            clsRegression = clsRegression[nonBG, indices[nonBG], ]
            predBBoxes = pred2bbox(clsRegression, roi[nonBG])
            mIoU = \
                getROILabel(predBBoxes, gtBBoxes[0,:, :4], gtBBoxes[0, :, -1], forMetric=True)
            losses2.append(mIoU)
            # precision
            precision = np.mean(np.squeeze(labels)==np.argmax(clsScores, axis=-1))
            losses2.append(precision)
            clsLoss += losses2

            if j%10==0:
                str1 = ''.join([printStyle.format(name, value) for name, value in zip(rpnMetricNames, losses1)])
                str2 = ''.join([printStyle.format(name, value) for name, value in zip(clsMetricNames, losses2)])
                print('epoch: %d, iteration: %d'%(e, j), str1, '', str2)
        clsLoss /= N
        rpnLoss /= N
        print('=========================================================================')
        print(clsLoss, rpnLoss)
        saveName = saveStyle.format(e, rpnLoss[-2], rpnLoss[-1], clsLoss[-2], clsLoss[-1])
        allModel.save_weights(os.path.join(save_path, saveName))

if __name__=='__main__':
    printStyle = '{0}:{1:.2f} '
    saveStyle = 'epoch:{0}_rpnMetric1:{1:.3f}_rpnMetric2:{2:.3f}_clsMetric1:{3:.3f}_clsMetric2:{4:.3f}'
    nb_classes = 21
    with open('./train.pickle', 'rb') as f:
        trainData = pickle.load(f)
    gen = Generator(trainData, [800, 600], 256, useAugmentation=False, angle=45, shift=10,
                 basesize=16, isSample=True,)
    img_input = Input(shape=[None, None, 3])
    roi_input = Input(shape=[None, 4 ])
    models = buildModel(img_input, roi_input, nb_classes, transfer=True)
    # epoch, init_epoch, topN, printStyle, saveStyle, loadPath
    train(models, gen, nb_classes, epoch=300, init_epoch=0, topN=120,
          printStyle=printStyle, saveStyle=saveStyle, lr=5e-6,
          loadPath= './allModels/epoch:33_rpnMetric1:0.984_rpnMetric2:0.447_clsMetric1:0.676_clsMetric2:0.625',
          save_path='./allModels/', freezeRPN=False, OnlyCls=False)
    #
    # './allModels/epoch:51_rpnMetric1:0.884_rpnMetric2:0.410_clsMetric1:0.527_clsMetric2:0.366'
    #['./models/epoch26[[0.728 0.367 0.089 0.91  0.437]].ckpt',
    #          '/home/johnsmith/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', ]