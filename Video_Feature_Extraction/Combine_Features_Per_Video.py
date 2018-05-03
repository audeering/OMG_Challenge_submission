'''

This file gathers all results created by the Multi_File_Extraction.py script
and merges them together in a single csv file.

For the code to run successfully, one must set the following variables:
- OMG_Data_Description_root: root directory of the omg_XXX.csv files
- results_root: directory in which to place the results

Copyright (c) audEERING GmbH, Gilching, Germany. All rights reserved.
http://www.audeeering.com/

Created: April 30 2018

Authors: Andreas Triantafyllopoulos, Hesam Sagha, Florian Eyben


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

import pandas as pd
import os
import numpy as np
from CSV_2_ARFF import CSV_2_ARFF
# change directories to local folders
OMG_Data_Description_root = '../../Data/OMGEmotionChallenge/'
train_sequence = OMG_Data_Description_root + 'omg_TrainVideos.csv'
validation_sequence = OMG_Data_Description_root + 'omg_ValidationVideos.csv'
test_sequence = OMG_Data_Description_root + 'omg_TestVideos_WithoutLabels.csv'
Network_name = 'VGGFace'
results_root = '/media/atriant/Secure/OMG/' + Network_name + '_Results/'

layer_name = 'f7' # change f5 to f6 f7 or f8 for respective VGGFace results
for file_sequence in [train_sequence, validation_sequence, test_sequence]:
    combined_features = pd.DataFrame()
    suffix = ''
    name_suffix = ''
    if (file_sequence == train_sequence):
        suffix = 'train_data/'
        name_suffix = 'Train'
    elif(file_sequence == validation_sequence):
        suffix = 'dev_data/'
        name_suffix = 'Dev'
    elif(file_sequence == test_sequence):
        suffix = 'test_data/'
        name_suffix = 'Test'
    results_dir = results_root + suffix
        
    with open(file_sequence) as f:
        next(f)
        current_video_sequence = ''
        segment_index = 0
        for l in f:
            l = l.strip()
            if len(l) > 0:
                
                video, utterance = l.split(',')[3:5]
                if file_sequence != test_sequence:
                    arousal, valence = l.split(',')[6:8]
                if video != current_video_sequence:
                    current_video_sequence = video
                    segment_index = 0
                    
                results_file = results_dir + video + '/' + utterance.split('.')[0] + '_' + layer_name + '_mean.csv'
                if not os.path.exists(results_file):
                    continue
                
                temp = pd.read_csv(results_file)
                temp = temp.drop('Unnamed: 0', axis = 1)
                filenames = [''] * len(temp)
                HasFace = [1] * len(temp)
                for index, row in temp.iterrows():
                    filenames[index] = video + '--' + "%.3d"%(segment_index)
                    if np.count_nonzero(row) == 0:
                        HasFace[index] = 0
                    segment_index = segment_index + 1
                
                if file_sequence != test_sequence:
                    temp['arousal'] = [arousal] * len(temp)
                    temp['valence'] = [valence] * len(temp)
#                temp['emotion'] = [emotion] * len(temp)
                temp['utterance'] = video + '/' + utterance
                temp['name'] = filenames
                temp['face'] = HasFace
                combined_features = combined_features.append(temp)
    
#    combined_features = combined_features.drop('Unnamed: 0', axis = 1)
    
    cols = combined_features.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    combined_features = combined_features[cols]
    csv_filename = results_root + Network_name + '_' + layer_name + '_' + name_suffix + '_' + 'Combined_mean.csv'
    combined_features.to_csv(csv_filename, index = False)
    arff_filename = results_root + Network_name + '_' + layer_name + '_' + name_suffix + '_' + 'Combined_mean.arff'
    CSV_2_ARFF(csv_filename, arff_filename, Network_name, layer_name)
