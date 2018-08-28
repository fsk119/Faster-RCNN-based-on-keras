import keras
from keras.layers import Conv2D, Activation, Input, MaxPooling2D, Reshape, Lambda
from keras import  Model
from keras import optimizers, regularizers
from vocDataGen import Generator
from RPN_feeder_loss import smooth_l1_loss, crossEntropy, recallMetric, wCEWrapper
from keras.applications.vgg16 import preprocess_input
from nms import ROILayer
import numpy as np
# utils for regression
# add regulaizer
def VGG16(img_input=None, trainable=False):
    if img_input is None:
        img_input = Input(shape=[None, None, 3])
    x = Lambda(lambda z: preprocess_input(z))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', )(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # kernel_regularizer=regularizers.l2(1e-3)
    return x

def RPN(baseLayer, numOfAnchors, ):
    intermediate = Conv2D(256, [3, 3],  kernel_initializer='normal', padding='same',
                          kernel_regularizer=regularizers.l2(1e-3))(baseLayer)
    # score
    score = Conv2D(numOfAnchors, [1, 1], kernel_regularizer=regularizers.l2(1e-3),
                   kernel_initializer='zero')(intermediate)
    score = Activation('sigmoid', name='scores')(score)

    # anchors regression
    anchors = Conv2D(numOfAnchors*4, [1, 1], name='anchors', kernel_regularizer=regularizers.l2(1e-3),
                     kernel_initializer='zero')(intermediate)
    return [score, anchors]

def buildRPNModel(transfer=True):
    input = Input([None, None, 3])
    encoder = VGG16(input, trainable=not transfer)
    score, anchors = RPN(encoder, 9)
    score = Reshape([-1, 1], name='trainScores')(score)
    anchors = Reshape([-1, 4], name='trainAnchors')(anchors)
    model = Model(inputs=input, outputs=[score, anchors])
    return model

def trainRPN(model, path, trainGen, **kwargs):
    if not path is None:
        import os
        if not os.path.exists(path):
            raise FileExistsError('%s doesn\'s exist'%path)
        model.load_weights(path, by_name=True)
    learning_rate = kwargs.get('lr', 1e-5)
    epoches = kwargs.get('epoches',1000)
    init_epoch = kwargs.get('init_epoch', 0)
    save_path = kwargs.get('save_path')
    validationGen = kwargs.get('validationGen', None)
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer, [wCEWrapper(2), smooth_l1_loss],loss_weights=[1, 4.0],
                  metrics={'trainScores':recallMetric})
    _callbacks = [] # later
    N = len(trainGen)
    roilayer = ROILayer('train', None)
    print('every epoch have %d steps'%N)
    if validationGen is None:
        M = 0
    else:
        M = len(validationGen)
    for e in range(init_epoch, epoches):
        currLoss = np.zeros([1, 4+1])
        for i in range(N):
            image, labels, targetBBoxes, gtBBoxes = next(trainGen)
            result = model.train_on_batch(image, [labels, targetBBoxes])
            predScores, predRegression = model.predict_on_batch(image)
            mIoU = roilayer(predScores, predRegression, trainGen.dims,
                        300, gtBBoxes=gtBBoxes[0, :, :4],
                        gtLabels=np.expand_dims(gtBBoxes[0, :, -1], axis=-1),
                        forMetric = True)
            result.append(mIoU)
            currLoss = currLoss+np.array(result)

            if i%10==0:
                print('epoch %d, iteration %d, '%(e, i), [(model.metrics_names[j], result[j])
                                                                 for j in range(4) ],
                      ', mIoU ', result[-1])
        currLoss /= N
        print('===============================================')
        print('epoch %d:'%e, currLoss)
        model.save_weights(save_path+'epoch%d'%e+np.array2string(currLoss[-2:],precision=3)+'.ckpt')

    return model


if __name__=='__main__':
    import pickle
    encoderPath = './models/epoch26[[0.728 0.367 0.089 0.91  0.437]].ckpt'#'/home/johnsmith/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    with open('./train.pickle', 'rb') as f:
        trainData = pickle.load(f)
    # with open('./validation.pickle', 'rb') as f:
        #validationData = pickle.load(f)
    trainGen = Generator(trainData, [800, 600], 256, useAugmentation=True, angle=45, shift=10,
                 basesize=16, isSample=True,)
    #trainValidation = Generator(validationData, [1000, 800], 32, useAugmentation=False, angle=45, shift=10,
                 #basesize=16, isSample=True,)
    model = buildRPNModel(transfer=True)
    trainRPN(model, encoderPath, trainGen, save_path='./models/',
             validationGen=None, init_epoch=29, lr=5e-5
             )
    pass