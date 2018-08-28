from PIL import Image
import numpy as np
from RPN_feeder_loss import genBaseAnchors, genAnchorLabel, pred2bbox

def resizeImage(image, bboxes, **kwargs):
    dims = np.array(kwargs.get('dims'))
    size = np.array(image.size)
    scales = dims/size
    scales = np.hstack([scales, scales])
    resizedBBoxes = np.round(bboxes*scales)
    resizedImage = image.resize(dims)
    return resizedImage, resizedBBoxes

def randomRotate(image, bboxes, parameters):
    angle = parameters.get('angle', 45)
    size = np.array(image.size)
    rangle = np.random.randint(-angle, angle, [1, ])
    rimage = image.rotate(rangle)
    rangle = -rangle/180.0*np.pi
    xmin, ymin, xmax, ymax = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    rotateMatrix = np.array([[np.cos(rangle), -np.sin(rangle)], [np.sin(rangle), np.cos(rangle)]])
    coordinates = np.stack([xmin, ymin, xmin, ymax, xmax, ymin, xmax, ymax], axis=1).\
        reshape([bboxes.shape[0], 4, 2])-size/2.0
    rcoordinates = np.matmul(coordinates, rotateMatrix.transpose())+size/2.0
    rmin, rmax = np.min(rcoordinates, axis=1), np.max(rcoordinates, axis=1)
    rbboxes = np.hstack([np.round(rmin), np.round(rmax)])
    # clip the value
    rbboxes[:, [0, 2]] = np.clip(rbboxes[:, [0, 2]], 0, size[0], )
    rbboxes[:, [1, 3]] = np.clip(rbboxes[:, [1, 3]], 0, size[1], )
    return rimage, rbboxes

def randomJitter(image, bboxes, parameters=None):
    shift = parameters.get('shift', 10)
    size = image.size
    x_shift = np.random.randint(-shift, shift, )
    y_shift = np.random.randint(-shift, shift)
    image = np.roll(image, [x_shift, y_shift], axis=(0, 1))
    shifts = np.hstack([y_shift, x_shift, y_shift, x_shift])
    bboxes = bboxes + shifts
    np.clip(bboxes[:, [0, 2]], 0, size[0], bboxes[:, [0, 2]])
    np.clip(bboxes[:, [1, 3]], 0, size[1], bboxes[:, [1, 3]])
    # convert ndarray to PIL.image
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    return image, bboxes

def randomClip(image, bboxes, parameters=None):
    isClip = np.random.uniform(0, 1.0)>=0.5
    size = np.array(image.size)
    if isClip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        bboxes[:, [0, 2]] = size[0]-bboxes[:, [2, 0]]
    isClip = np.random.uniform(0, 1.0)>=0.5
    if isClip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        bboxes[:, [1, 3]] = size[1]-bboxes[:, [3, 1]]
    return image, bboxes

# add some code for color adjustment later
def dataAugmentation(image, bboxes, parameters):
    allFunc = [randomClip, randomJitter, randomRotate]
    bboxes = bboxes.copy()
    for func in allFunc:
        image, bboxes = func(image, bboxes, parameters)
    return image, bboxes

#---------------------------------------------------------

class Generator:
    """
    1)data augmentation
    2)resolve voc dataset
    """
    def __init__(self, data, imgDims, batchsize, useAugmentation=False, angle=45, shift=10,
                 basesize=16, isSample=False, ratios=[0.5, 1.0, 2.0], scales=2**np.arange(3, 6)):
        self.data = data # data structure {image, bbox, class}
        self.dims = imgDims
        self.iter = 0
        self.indices = np.arange(len(data))
        self.batchsize = batchsize
        self.useAugmentation = useAugmentation
        if useAugmentation:
            self.parameters = {'angle': angle, 'shift':shift}
        self.baseAnchors = genBaseAnchors(basesize, ratios, scales)
        self.basesize = basesize
        self.isSample = isSample
        self.featureSize = [np.floor(dim/basesize).astype('int') for dim in self.dims]

    def __next__(self):
        file = self.data[self.indices[self.iter]]
        imagePath, bboxes = file['image'], file['bboxes'].copy()
        image = Image.open(imagePath)
        if self.useAugmentation:
            image, bboxes[:, :4] = dataAugmentation(image, bboxes[:, :4], self.parameters)
        image, bboxes[:, :4] = resizeImage(image, bboxes[:, :4], dims=self.dims)
        # genAnchorslabels

        labels, targetBBoxes = \
            genAnchorLabel(self.dims,
                           self.basesize, bboxes[:, :4], self.baseAnchors,
                           isSample=self.isSample,batchsize=self.batchsize,)
        self.iter = self.iter+1
        if self.iter==len(self.data):
            self.iter = 0
            self.on_epoch_end()
        return np.expand_dims(np.array(image), axis=0), np.expand_dims(labels, axis=0), \
               np.expand_dims(targetBBoxes, axis=0), np.expand_dims(bboxes, axis=0)

    def __len__(self):
        return len(self.data)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class kerasGeneratorWrapper:
    # maybe we can implement for ..
    pass

if __name__=='__main__':
    import pickle
    from visualization import visualization
    import matplotlib.pyplot as plt
    # test code for data augmentation
    with open('testGen.pickle', 'rb') as f:
        file = pickle.load(f)
    image = Image.open(file['image'])
    bboxes = file['bboxes'].copy()
    image, bboxes[:, :4] = resizeImage(image, bboxes[:, :4], dims=[800, 600])
    visImage = visualization(image, bboxes[:, :4], np.arange(bboxes.shape[0]))
    baseAnchors = genBaseAnchors(16, [0.5, 1.0, 2.0], 2**np.arange(3, 6))
    labels, targetBBoxes = genAnchorLabel([800, 600], 16, bboxes[:, :4], baseAnchors, batchsize=128)
    plt.imshow(visImage)
    plt.show()
    #parameters = {'angle':45, 'shift':10}
    #dataAugmentation(image, bboxes[:, :4], parameters)

    # test code for Generator v1(primary work)


    """
    gen = Generator(data, [800, 600], batchsize=64, isSample=True)
   
    sumP, sumN =0, 0
    for i in range(1000):
        image, labels, targetBBoxes, gtClsLabels = next(gen)
        positive = np.sum(labels[:, :, 1]==1)
        negative = np.sum(labels[:, :, 1]==0)
        sumP += positive
        sumN += negative
        print(positive, negative, np.sum(labels[:, :, 0]))
    print(sumP/sumN)
    # image, bboxes = next(gen)
   

    # test code for Generator v2(labels and  target bboxes)
    image, labels, targetBBoxes, bboxes = next(gen)
    # print(labels.shape, targetBBoxes.shape)
    indices = targetBBoxes[:, :, 0].astype('bool')
    _targetBBoxes = targetBBoxes[indices, 1:]
    # realBBoxes = pred2bbox(_targetBBoxes, positiveAnchors)
    # print(realBBoxes.shape, realBBoxes, positiveAnchors)

    # test code for nms
    from nms import getROI, nmsWrapper
    roi, scores, anchors = getROI(np.expand_dims(labels[:, :, -1], axis=-1), targetBBoxes[:, :, 1:],
                         gen.baseAnchors, 9, gen.dims,
                         gen.basesize, 20, nmsWrapper(0.7))
    print(roi.shape)
    print(roi)
    visImage = visualization(np.squeeze(image), roi, np.arange(roi.shape[0]))#?????
    plt.imshow(np.squeeze(visImage))
    plt.show()

 """