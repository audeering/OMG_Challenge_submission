'''
This code uses the MTCCN library for face detection and alignment, and
consecutively extracts deep features using VGGFace's fc7 layer. The features
are averaged over 2-second frame with an 1-sec increment.

Video files must be placed in the $database$ directory. OMG-Challenge
description files must be placed in the $OMG_Data_Description_root$ directory.

One csv file is created containing the features for each video, and is then
placed on a corresponding folder within the $results_root$ directory.

For the code to run successfully, one must set the following variables:
- OMG_Data_Description_root: root directory of the omg_XXX.csv files
- database: root directory of the OMG-Challenge raw video files
- results_root: directory in which to place the results
- caffe_root: directory of caffe installation
- caffe_model_path: directory of VGGFace and MTCCN caffe models


Copyright (c) audEERING GmbH, Gilching, Germany. All rights reserved.
http://www.audeeering.com/

Created: April XX 2018

Authors: Hesam Sagha, Andreas Triantafyllopoulos, Florian Eyben


Private, academic, and non-commercial use only!
See file "LICENSE.txt" for details on usage rights and licensing terms.
By using, copying, editing, compiling, modifying, reading, etc. this
file, you agree to the licensing terms in the file LICENSE.txt.
If you do not agree to the licensing terms,
you must immediately destroy all copies of this file.
 
THIS SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO EXPRESS,
IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT LIMITATION, WARRANTIES OF
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, ANY WARRANTY AGAINST
INTERFERENCE WITH YOUR ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE
OR NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL FULFILL
ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST PASS THIS
DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
NEITHER TUM NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE LIABLE FOR ANY
DAMAGES RELATED TO THE SOFTWARE OR THIS LICENSE AGREEMENT, INCLUDING
DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL DAMAGES, TO THE
MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT LEGAL THEORY IT IS BASED ON.
ALSO, YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE
THE SOFTWARE OR DERIVATIVE WORKS.
'''

import caffe
import numpy as np
import pandas as pd
import cv2
import skvideo.io
import os
from MTCCN import detect_face

if __name__ == '__main__':
    
    OMG_Data_Description_root = '../../Data/OMGEmotionChallenge/'
    train_sequence = OMG_Data_Description_root + 'omg_TrainVideos.csv'
    validation_sequence = OMG_Data_Description_root + 'omg_ValidationVideos.csv'
    test_sequence = OMG_Data_Description_root + 'omg_TestVideos_WithoutLabels.csv'
    database = ''
    results_root = ''
    # VGGFace initialization
    caffe.set_mode_gpu()
    caffe_root = ''
    caffe_model_path = "../../Data/DNN_Models/MTCNN/"
    net = caffe.Net(caffe_model_path + 'VGGFace/VGG_FACE_deploy.prototxt',
                    caffe_model_path + 'VGGFace/VGG_FACE.caffemodel',
                    caffe.TEST)
#    net = caffe.Net('../../Data/DNN_Models/DFL/dfl_deploy.prototxt',
#                '../../DFL/dfl_model.caffemodel',
#                caffe.TEST)
    #    net = caffe.Net('../../Data/DNN_Models/SphereFace/sphereface_deploy.prototxt',
#                '../../Data/DNN_Models/SphereFace/sphereface_model.caffemodel',
#                caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1, 3, 224, 224)
#    net.blobs['data'].reshape(1, 3, 112, 96)
   
    
    # MTCNN initialization
    minsize = 20
    
    threshold = [0.6, 0.7, 0.9]
    factor = 0.709
#    factor = 0.9 # set to 0.9 se we can get a single face
#    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path+"det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
    
    for file_sequence in [train_sequence, validation_sequence, test_sequence]:
        if (file_sequence == train_sequence):
            database_root = database + 'train_data/'
            directory = results_root + 'train_data/'
        elif(file_sequence == validation_sequence):
            database_root = database + 'dev_data/'
            directory = results_root + 'dev_data/'
        elif(file_sequence == test_sequence):
            database_root = database + 'test_data/'
            directory = results_root + 'test_data/'

        with open(file_sequence) as f:
            next(f)
            for l in f:
                l = l.strip()
                if len(l) > 0:
                    video, utterance = l.split(',')[3:5]
                    if os.path.exists(directory + video + '/' + utterance.split('.')[0] + '_' + 'f6_mean.csv'):
                        continue # if file has already been processed
                    print('Now processing:', video, utterance)
                    # Open Video
                    full_video_path = database_root + video + '/' + utterance
                    
                    vid = skvideo.io.vreader(full_video_path)
                    
                    metadata = skvideo.io.ffprobe(full_video_path)
                    if(len(metadata) == 0):
                        print ('File not found:', full_video_path, 'Skipping...')
                        continue
                    fps = int(metadata['video']['@r_frame_rate'].split('/')[0]) / int(metadata['video']['@r_frame_rate'].split('/')[1])
                    nFrames = int(metadata['video']['@nb_frames'])
                    nSegments = int(np.ceil(nFrames / fps))
                
                    column_names = [str(x) for x in range(len(net.blobs['fc7'].data[0]))]
                    f7_df = pd.DataFrame(data = [[0] * len(net.blobs['fc7'].data[0])] * nFrames, columns = column_names)
                    f7_df_seg_mean = pd.DataFrame(data = [[0] * len(net.blobs['fc7'].data[0])] * nSegments, columns = column_names)          
                    frame_counter = 0
                    for frame in vid:
                        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        
                        img_matlab = image.copy()
                        tmp = img_matlab[:,:,2].copy()
                        img_matlab[:,:,2] = img_matlab[:,:,0]
                        img_matlab[:,:,0] = tmp
                        
                        boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
                        if len(boundingboxes) > 0:
                            if type(boundingboxes[0]) == np.ndarray:
                                boundingboxes = boundingboxes[0]
                                
                            x = int(boundingboxes[0])
                            x = min(max(x, image.shape[1]), 0)
                            y = int(boundingboxes[1])
                            y = min(max(y, image.shape[0]), 0)
                            w = int(boundingboxes[2]) - x
                            h = int(boundingboxes[3]) - y
                        
                            im = image[y:y+h, x:x+w, (2,1,0)]
                            im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                            net.blobs['data'].data[...] = transformer.preprocess('data', im)
                        
                    
                            out = net.forward()
                            f7_df.loc[frame_counter] = net.blobs['fc7'].data[0]
                        else:
                            f7_df.loc[frame_counter] = np.zeros(len(net.blobs['fc7'].data[0]))
                            
                        frame_counter = frame_counter + 1
                    
                    for segment_index in range(nSegments):
                        segment_start = segment_index * fps
                        segment_end = (segment_index + 2) * fps - 1
                        f7_df_seg_mean.loc[segment_index] =  f7_df.loc[segment_start : segment_end].mean()
                        
                    
                    if not os.path.exists(directory + video + '/'):
                        os.makedirs(directory + video + '/')
                   f7_df_seg_mean.to_csv(directory + video + '/' + utterance.split('.')[0] + '_' + "f7_mean.csv")
