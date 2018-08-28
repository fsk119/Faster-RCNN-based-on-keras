import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
"""
There are 3 parts for resolving the dataset: 
1. grasp all the xml in the folder 
2. get the image path according to xml name
3. parse the xml and get the coordinates of bboxes
"""
mapper = {}
def name2label(name):
    if not mapper.get(name, None):
        cnt = len(mapper.keys())
        mapper[name] = cnt + 1
    return mapper[name]

def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name
    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise ValueError('illegal value for \'{}\': {}'.format(debug_name, e))
    return result

def _parse_annotation(element):
    """ Parse an annotation given an XML element.
    """
    truncated = _findNode(element, 'truncated', parse=int)
    difficult = _findNode(element, 'difficult', parse=int)
    class_name = _findNode(element, 'name').text

    box = np.zeros((1, 5))
    box[0, 4] = name2label(class_name)
    bndbox = _findNode(element, 'bndbox')
    box[0, 0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
    box[0, 1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
    box[0, 2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
    box[0, 3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1
    return truncated, difficult, box

def getBBoxes(root):
    bboxes = np.zeros((0, 5))
    for i, element in enumerate(root.iter('object')):
        truncated, difficult, box = _parse_annotation(element)
        if difficult:
            continue
        bboxes = np.append(bboxes, box, axis=0)
    return bboxes

def parseDataset(xmlRoot, imgRoot):
    xmlPathes = glob.glob(os.path.join(xmlRoot, '*.xml'))
    data = []
    for xmlPath in xmlPathes:
        name = os.path.split(xmlPath)[-1].split('.')[0]
        imgPath = os.path.join(imgRoot, name+'.jpg')
        if not os.path.exists(imgPath):
            continue
        root = ET.parse(xmlPath).getroot()
        bboxes = getBBoxes(root)
        if bboxes.shape[0]==0:
            continue
        data.append({'image': imgPath, 'bboxes':bboxes})
    return data

def splitData(dataset, ratio=0.2):
    train = []
    validation = []
    for data in dataset:
        prob = np.random.uniform(0, 1.0, 1)
        if prob<ratio:
            validation.append(data)
        else:
            train.append(data)
    return train, validation

#---------------------------------------------------------
if __name__=='__main__':
    import pickle
    data = parseDataset('./VOCdevkit/VOC2007/Annotations', './VOCdevkit/VOC2007/JPEGImages')
    with open('mapper.pickle', 'wb') as f:
        pickle.dump(mapper, f)
    print(len(data))
    train, validation = splitData(data)
    print(len(train), len(validation))
    with open('train.pickle', 'wb') as f:
        pickle.dump(train, f)
    with open('validation.pickle', 'wb') as f:
        pickle.dump(validation, f)