'''
This code is used to train a BiLSTM model on the deep face features and
make predictions on the official test set. 

Results are placed in Test_Results.csv file.

For the code to run successfully, one must set the following variables:
- OMG_Data_Description_root: root directory of the omg_XXX.csv files
- test_description_filename: path to the omg_TestVideos_WithoutLabels.csv
- features_root: directory of VGGFace features


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

from __future__ import print_function
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Masking
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from scipy.stats import pearsonr
from keras.models import load_model
#%% keras function definitions
def keras_loss(y_true, y_pred):
    return 1 - compute_ccc_keras(y_true, y_pred)
    
def compute_ccc_keras(y_true, y_pred):
    true_mean = K.mean(y_true)
    pred_mean = K.mean(y_pred)
    fsp = y_pred - pred_mean
    fst = y_true - true_mean

    devP = K.std(y_pred) + K.epsilon()
    devT = K.std(y_true) + K.epsilon()

    rho = (K.mean(fsp * fst)) / (devP * devT)
    ccc = 2 * rho * devT * devP / (
        devP * devP + devT * devT +
        (pred_mean - true_mean) * (pred_mean - true_mean) + K.epsilon())

    return (ccc)
#%% final evaluation metric
def compute_ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)

    rho,_ = pearsonr(y_pred,y_true)

    std_predictions = np.std(y_pred)

    std_gt = np.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)

    return ccc, rho
#%% feature normalization
def normalize_features(df_train, df_dev, feature_names):
    scaler = StandardScaler()
    df_train.loc[:, feature_names] = scaler.fit_transform(df_train[feature_names].values)
    df_dev.loc[:, feature_names] = scaler.transform(df_dev[feature_names].values)
    
    return df_train, df_dev, scaler
#%% create stateless LSTM sequences  
def create_stateless_sequence(df, targ, feature_names, key = 'utterance'):
    unique_sequences = df[key].unique()
    features = [[0] * len(feature_names)] * len(unique_sequences)
    labels = [0] * len(unique_sequences)
    for index in range(len(unique_sequences)):
        keyVal = unique_sequences[index]
        features[index] = df.loc[df[key] == keyVal, feature_names].values
        if targ != '':
            labels[index] = df.loc[df[key] == keyVal, targ].values[0]
        
    return features, labels

def create_stateless_LSTM_sequence(df, pad_value, max_seq_len, feature_names, targ = 'valence', key = 'utterance'):

    features, labels = create_stateless_sequence(df, targ, feature_names, key)
    x = pad_sequences(features, dtype = 'float64', maxlen = max_seq_len, value = pad_value)
    y = np.array(labels)
    
    return x, y

#%% LSTM model for OMG seq2seq regression
def create_LSTM_train_model(pad_value, max_seq_len, number_of_features):
    model = Sequential()
    model.add(Masking(mask_value = pad_value,
                            input_shape=(max_seq_len, number_of_features)))
    model.add(Bidirectional(LSTM(16, return_sequences = True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(8)))
#    model.add(Bidirectional(LSTM(8)))
    model.add(Dense(1, activation = 'tanh'))
    model.compile('adam', loss = keras_loss)
    return model

def create_LSTM_model(pad_value, max_seq_len, number_of_features, batch_size):
    model = Sequential()
    model.add(Masking(mask_value = pad_value,
                            input_shape=(max_seq_len, number_of_features),
                                 batch_input_shape = (batch_size, max_seq_len, number_of_features)))
    model.add(Bidirectional(LSTM(16, return_sequences = True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(8)))
#    model.add(Bidirectional(LSTM(8)))
    model.add(Dense(1, activation = 'tanh'))
    model.compile('adam', loss = keras_loss)
    return model

#%% main function
if __name__ == '__main__':
    
    #%% specify feature set
    features_root = '../../Data/Video_Features/'
    network = 'VGGFace' # use VGGFace Deep Features
    layer = 'f7' # layers to use as features
    network_root = features_root + network + '/' 
    #%% load training and validation set
     
    train_filename = network_root + network + '_' + layer +'_Train_Combined_extended.csv'
    df_train = pd.read_csv(train_filename)
    dev_filename = network_root + network + '_' + layer +'_Dev_Combined_extended.csv'
    df_dev = pd.read_csv(dev_filename)
    
    #%% add utterance information
    train_utterance = pd.read_csv('../../Data/Masks/Sample_to_Utterence_Train.csv')
    dev_utterance = pd.read_csv('../../Data/Masks/Sample_to_Utterence_Dev.csv')
    
    df_train['utterance'] = train_utterance['0'].values
    df_dev['utterance'] = dev_utterance['0'].values

    #%% create video feature
    df_train['video'] = [x.split('--')[0] for x in df_train['name']]    
    df_dev['video'] = [x.split('--')[0] for x in df_dev['name']]
    
    #%% discard segments when no face is detected
    df_train = df_train[df_train['face'] == 1]
    df_dev = df_dev[df_dev['face'] == 1]
    
    #%% merge original training and validation sets
    df_train = df_train.append(df_dev).reset_index(drop = True)
    del df_dev
    #%% experiment parameters
    feature_names = [x for x in df_train.columns if x not in ['video', 'name', 'utterance', 'arousal', 'valence', 'emotion', 'face']]  
    pad_value = -10 # value to mask in missing timesteps
    targ = 'valence' # attribute to predict
    max_seq_len = max(Counter(df_train['utterance'].values).values())
    key = 'utterance'
    epochs = 20
    folds = 6
    number_of_features = len(feature_names)
    train_batch = 64
    
    s = list(range(len(df_train)))
    fold_size = int(len(df_train) / folds)
    transformations = [StandardScaler()] * folds
        
    #%% model training
    for fold in range(folds):
        model_name = 'RndSkr_fold_' + str(fold+1) + '.h5'
        fold_dev_indices = s[fold * fold_size : (fold + 1) * fold_size]
        fold_train_indices = [index for index in range(len(df_train)) if index not in fold_dev_indices]

        train_df = df_train.iloc[fold_train_indices, :].reset_index(drop = True)
        dev_df = df_train.iloc[fold_dev_indices, :].reset_index(drop = True)
        
        train_df, dev_df, scaler = normalize_features(train_df, dev_df, feature_names)
        transformations[fold] = scaler
        x_train, y_train = create_stateless_LSTM_sequence(train_df, pad_value, max_seq_len, feature_names, targ, key)
        x_dev, y_dev = create_stateless_LSTM_sequence(dev_df, pad_value, max_seq_len, feature_names, targ, key)
       
        model = create_LSTM_train_model(pad_value, max_seq_len, number_of_features)
        indices = [index for index in range(len(y_dev)) if y_dev[index] != pad_value]
        x = y_dev[indices]    
        max_ccc = -10
        best_epoch = 0
        dev_batch = x_dev.shape[0]
        for epoch in range(epochs):
            model.fit(x_train, y_train, epochs = 1, batch_size = train_batch, verbose = 0)
            y_pred = model.predict(x_dev, batch_size = dev_batch)
            ccc,_ = compute_ccc(y_dev, y_pred[:, 0])
            # early stopping based on ccc of out-of-fold set
            if max_ccc < ccc:
                best_epoch = epoch
                max_ccc = ccc
                model.save(model_name)
                
        print('Best epoch for fold #', str(fold+1), ' was epoch #', str(best_epoch+1), ' with a ccc of:', max_ccc)

    #%% re-load test set (ram issues)
    del df_train, x_train, y_train, x_dev, y_dev
    test_filename = network_root + network + '_' + layer +'_Test_Combined.csv'
    df_test = pd.read_csv(test_filename)
    
    df_test['utterance'] = [x.split('.')[0] for x in df_test['utterance']] 
    
    df_test = df_test.dropna()
    #%% prepare results dataframe
    df_results = pd.DataFrame(df_test['utterance'].unique(), columns = ['name'])  
    
    df_results['utterance'] =  [x.split('/')[1] for x in df_results['name']]
    df_results['name'] =  [x.split('/')[0] for x in df_results['name']]
    df_results['valence'] = 0
    #%% use model ensemble to make predictions on the test set
    y_pred = np.zeros((len(df_test['utterance'].unique()), folds))
    for fold in range(folds):
        df_test_local = df_test.copy()
        model_name = 'RndSkr_fold_' + str(fold+1) + '.h5' #current model name
        model = load_model(model_name, custom_objects={'keras_loss': keras_loss}) #load model
        scaler = transformations[fold] #load correct feature transformation
        df_test_local.loc[:, feature_names] = scaler.transform(df_test_local[feature_names].values) #transform features
        x_test, _ = create_stateless_LSTM_sequence(df_test_local, pad_value, max_seq_len, feature_names, '', key)
        test_batch = 64
        predictions = model.predict(x_test, batch_size = test_batch)
        y_pred[:, fold] = predictions[:, 0]
    avg_pred = np.mean(y_pred, axis = 1)
    
    df_results['valence'] = avg_pred
    #%% replicate missing values
    df_results_temp = df_results.copy()
    test_description = pd.read_csv('../../Data/OMGEmotionChallenge/omg_TestVideos_WithoutLabels.csv')
    df_results_temp['utterance'] = [x + '.mp4' for x in df_results_temp['utterance']]

    for index in range(len(test_description)):
        if not (df_results_temp.loc[index, 'name'] == test_description.loc[index, 'video'] and \
                df_results_temp.loc[index, 'utterance'] == test_description.loc[index, 'utterance']):
                    new_element = pd.DataFrame([], columns = df_results_temp.columns)
                    new_element.loc[0, :] = [test_description.loc[index, 'video'], 
                                               test_description.loc[index, 'utterance'],
                                               df_results_temp.loc[index, 'valence']]
                    
                    df_results_temp = pd.concat([df_results_temp.loc[:index-1, :], new_element, 
                                                 df_results_temp.loc[index:, :]]).reset_index(drop = True)
    
    df_results_temp.to_csv('Test_Results.csv', index = False)
    
