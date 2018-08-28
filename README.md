# Faster-RCNN-based-on-keras

# Introduction
My idea is from https://github.com/jinfagang/keras_frcnn. In this repository, The structure is vgg16-->rpn proposal-->cls and regression net. 
Because rpn proposal get topN roi according to sorted rpn scores, I decide to implement with numpy for convenience. As a result, it much slow comparing to original implementation. It almost takes half an hour to train 4000 images in one epoch.

# Requirment
keras version is 2.1.4
tensorflow version is 1.4.0

# Attention
One thing needs to pay attention: I modify part of keras files. In my implementation, I use keras.Model.train_on_batch(...) to train net and in this function, it will use another function _standardize_user_data to _check_array_lengths. It ask the output of net have the same batchsize as input, but we have to crop image from input image according to batched roi. So I just turn off _check_array_lengths. It is very easy to modify, just set check_array_lengths = False in head of _standardize_user_data function.

# Something More
I have to admit that I haven't fine tune my network, but in some case, it has already worked pretty well and you can check testFinal.ipynb by yourselves.

It's my honour to help you if you have any questions and you can connect me through fsk119@zju.edu.cn
