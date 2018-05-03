'''
This code has been submitted to One-Minute-Gradual Emotion Challenge.
It extracts the audio features from the wav files using openSMILE toolkit.

Copyright (c) audEERING GmbH, Gilching, Germany. All rights reserved.
http://www.audeeering.com/

Created: April 30 2018

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
from subprocess import call
import pandas as pd
import os

fold = 'train'
feat_list = 'comp16'
hasTest = True
# TODO: set this folder to the directory in which
#	test_data,train_data,dev_data folders and
# 	omg_TestVideos_WithoutLabels.csv,omg_ValidationVideos.csv,omg_TrainVideos.csv files
#	are located
data_folder = ''

if feat_list == 'comp6000':
    config_file = 'config/ComParE_2016.conf'
elif feat_list == 'comp16':
    config_file = 'config/ComParE_2016_basicFuncs.conf'
elif feat_list == 'gemaps':
    config_file = 'config/gemaps/eGeMAPSv01a.conf'

if fold == 'test':
    fold_name = 'test_data'
    video_csv_file = data_folder + '../omg_TestVideos_WithoutLabels.csv'
elif fold == 'dev':
    fold_name = 'dev_data'
    video_csv_file = data_folder + '../omg_ValidationVideos.csv'
elif fold == 'train':
    fold_name = 'train_data'
    video_csv_file = data_folder + '../omg_TrainVideos.csv'

no_file = []
with open(video_csv_file) as csvfile:
    rows = pd.read_csv(csvfile)
    for ind, row in rows.iterrows():
        wav_file = data_folder + fold_name + "/" + row['video'] + "/" + row['utterance'] + ".wav"
        if not os.path.isfile(wav_file):
            print('not found:' + wav_file)
            no_file.append(wav_file)
            continue
        if len(row) == 8:
            arousal = str(row['arousal'])
            valence = str(row['valence'])
        else:  # test file has no label
            arousal = '0.0'
            valence = '0.0'
        call(['./SMILExtract',
              '-I', wav_file,
              '-C', config_file,
              '-O', 'extracted_features/feat_' + feat_list + '_' + fold_name + ".arff",
              '-instname', row['video'] + "/" + row['utterance'].split('.')[-2],
              '-arousal', arousal,
              '-valence', valence,
              '-arfftargetsfile', "config/shared/arff_targets.conf.inc",
              '-l', '1'
              ])
print('not existing files:' + no_file)

fname_feat_dev = 'extracted_features/feat_' + feat_list + '_dev_data.arff'
fname_feat_train = 'extracted_features/feat_' + feat_list + '_train_data.arff'
fname_out = 'extracted_features/feat_allmixed_' + feat_list + '.arff'
if hasTest:
    file_order = pd.read_csv('New_FileOrder_5mixed_trdevts.csv')
    fname_feat_test = 'extracted_features/feat_' + feat_list + '_test_data.arff'
    fname_out = 'extracted_features/feat_allmixed_' + feat_list + '_trdevts_data.arff'
else:
    file_order = pd.read_csv('../New_FileOrder_5mixed_trdev.csv')

with open(fname_feat_train, 'rt') as fp:
    feat_train = arff.load(fp)
with open(fname_feat_dev, 'rt') as fp:
    feat_dev = arff.load(fp)
if hasTest:
    with open(fname_feat_test, 'rt') as fp:
        feat_test = arff.load(fp)

arff_new = {'data': [], 'attributes': feat_dev['attributes'], 'relation': 'moved2new folds'}
feat_train_name = np.array([x[0] for x in feat_train['data']])
feat_dev_name = np.array([x[0] for x in feat_dev['data']])
if hasTest:
    feat_test_name = np.array([x[0] for x in feat_test['data']])

for f in file_order.values:
    spk_name = f[0]
    vid_name = f[1]
    utt_name = f[2]
    fnew = vid_name + '/' + utt_name.split('.')[0]  # remove .mp4
    fromFold = 'tr'
    idx = np.nonzero(feat_train_name == fnew)[0]
    if len(idx) == 0:
        fromFold = 'dev'
        idx = np.nonzero(feat_dev_name == fnew)[0]
        if hasTest:
            if len(idx) == 0:
                fromFold = 'test'
                idx = np.nonzero(feat_test_name == fnew)[0]
    if len(idx) == 0:
        print('NOT FOUND:' + fnew)
        continue
    x = []
    if fromFold == 'tr':
        for id in idx:
            x.append(feat_train['data'][id])
    elif fromFold == 'dev':
        for id in idx:
            x.append(feat_dev['data'][id])
    elif fromFold == 'test':
        for id in idx:
            x.append(feat_test['data'][id])
    for i in range(len(x)):
        x[i][0] = spk_name + '--' + str(i).rjust(4, '0') + '.' + vid_name + '/' + utt_name
    arff_new['data'] += x.copy()


def dump2arffnc(data, fname):
    arff.dump(data, open(fname, 'wt'))
    call(['./fixarfffiles.sh', fname])
    call(['./arff2nc_noorder', fname, '2', '2', fname + '.nc'])


dump2arffnc(arff_new, fname_out)
