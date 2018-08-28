from keras.engine.topology import Layer
from keras.layers import Input, Lambda
from keras import Model
import keras.backend as K
import tensorflow as tf
import numpy as np
"""
1. through roi and features pool and resize roi
2. build network
"""
class RPNPooling(Layer):
    def __init__(self, pool_size, **kwargs):
        super(RPNPooling, self).__init__(**kwargs)
        self.pool_size = pool_size

    def build(self, input_shape):
        super(RPNPooling, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        featureShape, roiShape = input_shape
        dims = None if roiShape[0] is None or roiShape[1] is None else roiShape[0]*roiShape[1]
        return (roiShape[1], self.pool_size, self.pool_size, featureShape[-1])

    def call(self, inputs, **kwargs):
        assert len(inputs) == 2
        features, roi = inputs
        """
        we need to transform style of the roi
        origin shape of roi: imgBatch x numBBox x 4
        we cancle the axis of imgBatch, and get (imgBatch x numBBox) x 5
        """
        roi = K.cast(roi, 'int32')
        inputDims = K.shape(roi)
        imgDims = K.expand_dims(K.arange(0, inputDims[0]), axis=1)
        roiDims = K.ones([1, inputDims[1]], dtype='int32')
        dims = K.expand_dims(imgDims * roiDims, axis=-1)
        _rois = K.concatenate([dims, roi], axis=-1)
        _rois = K.reshape(_rois, [-1, 5])
        def _stepFunction(unused, bbox):
            imageDim, y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
            roi = features[imageDim,x1:x2, y1:y2, :]
            resizedROI = tf.image.resize_images(roi, (self.pool_size, self.pool_size))
            return resizedROI
        result = tf.scan(_stepFunction, _rois, initializer=K.zeros([self.pool_size, self.pool_size, inputDims[-1]]))
        return result

def clsCEWrapper(nb_classes):
    def clsCE(y_true, y_pred):
        y_true = K.cast(y_true, 'int32')
        y_true = K.one_hot(y_true, nb_classes)
        return K.mean(K.categorical_crossentropy(y_true, y_pred))
    return clsCE
"""
idea is from https://github.com/meetshah1995/pytorch-semseg/issues/46
weights = [0.01571903, 0.99046264, 0.23850667, 1.38825576, 1.71287941,
       0.65848637, 1.21788251, 0.84606811, 1.        , 0.81868283,
       1.8857879 , 3.77297837, 1.25280998, 0.79205693, 0.6538707 ,
       1.60683092, 0.71810243, 1.8894318 , 0.42370746, 1.28170897,
       1.26907871]
       
"""
def clsWeigtedCEWrapper(nb_classes, weights):
    weights = K.variable(weights)
    def weightedCE(y_true, y_pred):
        y_true = K.cast(y_true, 'int32')
        sliceWeight = tf.gather_nd(weights, y_true)
        sliceWeight = K.reshape(sliceWeight, [-1, 1])
        y_true = K.one_hot(y_true, nb_classes)
        ce = K.categorical_crossentropy(y_true, y_pred)
        res =  K.mean(ce*sliceWeight)
        print(K.ndim(res), res.shape, ce.shape)
        return res
    return weightedCE

def clsSLWrapper(nb_classes):
    def clsSmoothL1(y_true, y_pred):
        indices = K.expand_dims(K.cast(y_true[:, 0], 'int32'), axis=-1)
        isValid = K.repeat_elements(K.cast(indices>0, 'float32'), 4, axis=-1)
        y_pred = K.reshape(y_pred, [-1, nb_classes, 4])
        N = K.shape(y_true)[0]
        imgDim = K.expand_dims(K.arange(0, N, dtype='int32'), axis=-1)
        #y_pred = y_pred[imgDim, indices, ]

        indices = K.concatenate([imgDim, indices], axis=-1)
        y_pred = tf.gather_nd(y_pred, indices)
        d = K.abs(y_pred-y_true[:, 1:])
        isLess = K.cast(d<1.0, 'float32')
        left = 0.5*isLess*K.square(d*isValid)
        right = (1-isLess)*((d-0.5)*isValid)
        return K.sum(left+right)/(K.sum(isValid)+K.epsilon())
    return clsSmoothL1

def mAPMetricWrapper(nb_classes, k=11):
    # idea is from https://github.com/broadinstitute/keras-rcnn/issues/6
    def mAPMetric(y_true, y_pred):
        # one-hot encoding
        y_true = K.squeeze(K.cast(y_true, 'int64'), axis=1)
        y_true = K.cast(K.one_hot(y_true, nb_classes), 'int64')
        _, batchmAP = tf.metrics.sparse_average_precision_at_k(y_true, y_pred, k)
        K.get_session().run(tf.local_variables_initializer())
        return batchmAP
    return mAPMetric

if __name__=='__main__':
    # get runtime batchsize
    """
    import numpy as np
    x = Input([None, 4])
    y = testLayer()(x)
    model = Model(x, y)
    z = model.predict_on_batch(np.random.rand(3, 5, 4))
    print(z)
   
    #-----------------------------------------------
    # test RPNPooling
    import numpy as np

    bboxes = Input([None, 4])
    features = Input([None, None, 1])
    pooledROI = RPNPooling(3)([features, bboxes])
    model = Model([features, bboxes], pooledROI)
    haha = np.array([[2, 4, 7, 9], [1, 4, 9, 9]])
    image = np.random.rand(1, 10, 10, 1)
    print(haha)
    z = model.predict_on_batch([image, np.expand_dims(haha, axis=0)])
    print(z.shape)
     """
    #-------------------------------------------------
    # test using real image
    from vocDataGen import Generator
    import pickle
    import numpy as np
    with open('./train.pickle', 'rb') as f:
        data = pickle.load(f)
    gen = Generator(data, [1000, 800], batchsize=32, isSample=True)
    image, labels, targetBBoxes, gtClsLabels = next(gen)

    from nms import getROI, nmsWrapper
    roi, scores, anchors = getROI(np.expand_dims(labels[:, :, -1], axis=-1), targetBBoxes[:, :, 1:],
                                  gen.baseAnchors, 9, gen.dims,
                                  gen.basesize, 5, nmsWrapper(0.7))
    print(roi, image.shape)
    bboxes = Input([None, 4])
    features = Input([None, None, 3])
    pooledROI = RPNPooling(400)([features, bboxes])
    model = Model([features, bboxes], pooledROI)
    roiImage = model.predict_on_batch([image, np.expand_dims(roi, axis=0)])
    import matplotlib.pyplot as plt

    for i in range(3):
        fig = plt.figure()
        plt.imshow(roiImage[i, ].astype('uint8'))
    plt.show()

    #print(roiImage[0, ].shape)