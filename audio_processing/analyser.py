'''
This code has been submitted to One-Minute-Gradual Emotion Challenge.
It performs the analysis on the extracted audio features. The regressor
is based on the CURRENNT toolkit.

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

from scipy.stats import pearsonr
import pandas as pd
import arff
import numpy as np
from scipy.stats import norm
from subprocess import call
import copy
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
import gc

feat_set = 'comp16'
hasTest = True
save_last_fold_result = True
if hasTest:
    folding_file = '../New_FileOrder_mixed_5trdev1origts.csv'
    input_feat_file = 'extracted_features/feat_allmixed_' + feat_set + '_trdevts_data.arff'
    save_result_file = 'final_result_arousal_valence_7trdev_1ts_shuffled_data.csv'
    orig_test_order = pd.read_csv('../omg_TestVideos_WithoutLabels.csv')
else:
    folding_file = '../New_FileOrder_5tr1origDev.csv'
    input_feat_file = 'extracted_features/feat_allmixed_' + feat_set + '.arff'
    save_result_file = 'final_result_dev.csv'
    orig_test_order = pd.read_csv('../omg_ValidationVideos.csv')
classifier = {'options_file': 'my_currennt.config',
              'network': 'currennt-config/net_proto02b_negtccc.jsn'}
normalization = {'method': 'cdf', 'locality': 'global', '2d': False}
nTarget = 2
targetsToPredict = [0]  # , 1]
targetNormalize = [True]  # , False]
metrics = ['cc', 'ccc', 'ccc_norm']
tmp_folder = 'tmp/'
test_on_last_fold = True
shuffle_training_data = True
shuffle_spk_balanced = False
np.random.seed(20)


def get_sequences(fnames, targets=None):
    names = np.array([xx[0].split('--')[0] + "/" + xx[0].split('.')[1] for xx in fnames])
    unique_names = np.unique(names)
    seqs = np.zeros(fnames.shape)
    for ii in range(len(unique_names)):
        seqs[names == unique_names[ii], 0] = ii
    if targets is not None:
        prev_name = ""
        k = 0
        target = np.zeros((len(unique_names), targets.shape[1]))
        for ii in range(len(names)):
            if prev_name != names[ii]:
                target[k] = targets[ii, :]
                prev_name = names[ii]
                k += 1
    else:
        target = targets
    return seqs, target


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def myprint(text, color=Bcolors.HEADER):
    print(color + text + Bcolors.ENDC)


def load_data_into_folds():
    myprint('Loading to folds...', Bcolors.OKGREEN)
    file_order = pd.read_csv(folding_file)
    nfold_original = np.max(file_order['fold'])
    if shuffle_training_data or shuffle_spk_balanced:
        final_folds = 3
    else:
        final_folds = nfold_original
    nfold = final_folds
    name_fold_dict = {}
    np.random.seed(20)
    if shuffle_spk_balanced:
        if test_on_last_fold:
            skf = StratifiedKFold(n_splits=final_folds - 1)
        else:
            skf = StratifiedKFold(n_splits=final_folds)
        n_split = skf.get_n_splits(file_order.values, file_order['rootfolder'])
        balanced_test_folds = []
        for _, test_folds in skf.split(file_order.values, file_order['rootfolder'].values):
            balanced_test_folds.append(test_folds)
        for indd, f in file_order.iterrows():
            rand_fold = 0
            if test_on_last_fold:
                if f['fold'] == nfold_original:
                    rand_fold = final_folds - 1
                else:
                    for ii in range(n_split):
                        if indd in balanced_test_folds[ii]:
                            rand_fold = ii
                            break
            else:
                for ii in range(n_split):
                    if indd in balanced_test_folds[ii]:
                        rand_fold = ii
                        break
            name_fold_dict[f['rootfolder'] + '/' + f['video'] + '/' + f['utterance']] = rand_fold
    else:
        for _, f in file_order.iterrows():
            if shuffle_training_data:
                if test_on_last_fold:
                    if f['fold'] == nfold_original:
                        rand_fold = final_folds
                    else:
                        rand_fold = np.random.randint(1, final_folds)
                else:
                    rand_fold = np.random.randint(1, final_folds + 1)
                name_fold_dict[f['rootfolder'] + '/' + f['video'] + '/' + f['utterance']] = rand_fold - 1
            else:
                name_fold_dict[f['rootfolder'] + '/' + f['video'] + '/' + f['utterance']] = f['fold'] - 1

    with open(input_feat_file, 'rt') as fp:
        data = arff.load(fp)
    att_names = data['attributes'][1:-nTarget]
    targ_names = np.array(data['attributes'][-nTarget:])[:, 0]
    data_folded = []
    for ifold in range(nfold):
        data_folded.append({'fname': [], 'data': [], 'target': [], 'seq': [], 'spk': [], 'utterance': []})
    for d in data['data']:
        spk_name = d[0].split('--')[0]
        vid_name = d[0].split('--')[1].split('.')[1].split('/')[0]
        utt_name = d[0].split('--')[1].split('/')[1]
        fold = name_fold_dict[spk_name + '/' + vid_name + '/' + utt_name]
        data_folded[fold]['fname'].append(d[0])
        data_folded[fold]['spk'].append(spk_name)
        data_folded[fold]['utterance'].append(utt_name)
        data_folded[fold]['data'].append(d[1:-nTarget])
        data_folded[fold]['target'].append(d[-nTarget:])
    for ifold in range(nfold):
        data_folded[ifold]['fname'] = np.array(data_folded[ifold]['fname']).reshape(-1, 1)
        data_folded[ifold]['spk'] = np.array(data_folded[ifold]['spk']).reshape(-1, 1)
        data_folded[ifold]['utterance'] = np.array(data_folded[ifold]['utterance']).reshape(-1, 1)
        data_folded[ifold]['data'] = np.array(data_folded[ifold]['data'])
        data_folded[ifold]['target'] = np.array(data_folded[ifold]['target'])
        data_folded[ifold]['target'] = data_folded[ifold]['target'][:, targetsToPredict]
    # normalizing by * 2 -1
    for ifold in range(nfold):
        for iTarget in range(len(targetsToPredict)):
            if targetNormalize[iTarget]:
                data_folded[ifold]['target'][:, iTarget] = data_folded[ifold]['target'][:, iTarget] * 2 - 1
    # extracting seqences
    for ifold in range(nfold):
        data_folded[ifold]['seq'], data_folded[ifold]['target'] = get_sequences(data_folded[ifold]['fname'],
                                                                                data_folded[ifold]['target'])
    targ_names = targ_names[targetsToPredict]
    return data_folded, targ_names, att_names


def prepare_data(data, test_fold, dev_fold):
    myprint('Preparing folds...', Bcolors.OKGREEN)
    nfold = len(data)
    train_folds = [xx for xx in range(nfold) if xx != test_fold and xx != dev_fold]

    dev_data = data[dev_fold]
    test_data = data[test_fold]
    train_data = copy.deepcopy(data[train_folds[0]])
    for f in train_folds[1:]:
        train_data['data'] = np.vstack((train_data['data'], data[f]['data']))
        train_data['fname'] = np.vstack((train_data['fname'], data[f]['fname']))
        train_data['target'] = np.vstack((train_data['target'], data[f]['target']))
        train_data['spk'] = np.vstack((train_data['spk'], data[f]['spk']))
        train_data['utterance'] = np.vstack((train_data['utterance'], data[f]['utterance']))

    train_data['seq'], _ = get_sequences(train_data['fname'], None)
    return train_data, dev_data, test_data


def normalize_target(data):
    all_samples = data[normalization['locality']]
    unique_samples = np.unique(all_samples)
    exp_lab = expand_labels(data).reshape(-1, len(targetsToPredict))
    for samp in unique_samples:
        temp = exp_lab[(samp == data[normalization['locality']]).flatten(), :]
        temp = (temp - np.mean(temp, axis=0)) / np.std(temp, axis=0)
        exp_lab[(samp == data[normalization['locality']]).flatten()] = temp
    temp = compress_labels(exp_lab, data)
    return temp


def normalize_data(data, argss):
    myprint('Normalising features...', Bcolors.OKGREEN)
    if normalization['method'] == 'meanstd':
        if normalization['locality'] == 'global':
            if argss is None:
                argss = {'mean': np.mean(data['data'], axis=0), 'std': np.std(data['data'], axis=0)}
            data['data'] = (data['data'] - argss['mean']) / argss['std']
        else:
            all_samples = data[normalization['locality']]
            unique_samples = np.unique(all_samples)
            for samp in unique_samples:
                new_d = data['data'][(samp == all_samples).flatten(), :]
                normed = (new_d - np.mean(new_d)) / np.std(new_d)
                data['data'][(samp == all_samples).flatten(), :] = normed
            data['target'] = normalize_target(data)
    if normalization['method'] == 'cdf':
        if normalization['locality'] == 'global':
            unique_samples = np.zeros((1, 1))
            all_samples = np.zeros((data['data'].shape[0], 1))
        else:
            all_samples = data[normalization['locality']]
            unique_samples = np.unique(all_samples)
            data['target'] = normalize_target(data)

        for samp in unique_samples:
            new_d = data['data'][(samp == all_samples).flatten(), :]
            n_s, n_f = new_d.shape
            resolution = 3000
            invcdf = np.linspace(-16, 16, resolution)
            nrm = norm.cdf(invcdf).reshape(1, -1)
            norm_rep = np.repeat(nrm, n_s, axis=0)

            cu = [xx / (n_s - 1) for xx in range(n_s)]
            cu = np.array(cu).reshape(-1, 1)
            diffs = np.abs(cu - norm_rep)
            indices = np.argmin(diffs, axis=1)
            new_vals = invcdf[indices]  # assiging to the sorted values

            for fi in range(n_f):
                vals = new_d[:, fi]  # data['data'][:, fi]
                sorted_args = np.argsort(vals)
                vals = np.reshape(vals, (1, -1))
                vals[0, sorted_args] = new_vals
                # ALREADY DONE mixed[:, fi] = vals[0, :]
            data['data'][(samp == all_samples).flatten(), :] = new_d
    return data, argss


def ccc_metric(y_true, y_pred):
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    rho, _ = pearsonr(y_pred, y_true)
    std_predictions = np.std(y_pred)
    std_gt = np.std(y_true)
    ccc_val = 2 * rho * std_gt * std_predictions / (
            std_predictions ** 2 + std_gt ** 2 +
            (pred_mean - true_mean) ** 2)
    return rho, ccc_val


def normalize_predictions(y_pred, mean_tr, std_tr):
    return (y_pred - np.mean(y_pred, axis=0)) / np.std(y_pred, axis=0) * std_tr + mean_tr


def calculate_accuracy(string, y_true, y_pred, mean_tr=None, std_tr=None):
    all_vals = {}
    if mean_tr is None:
        mean_tr = np.mean(y_true, axis=0)
        std_tr = np.std(y_true, axis=0)
    for metric in metrics:
        all_vals[metric] = []
    for target in range(y_true.shape[1]):
        mystr = string + " " + target_names[target] + ": "
        val = []
        for metric in metrics:
            if metric == 'cc':
                val, _ = ccc_metric(y_true[:, target], y_pred[:, target])
            if metric == 'ccc':
                _, val = ccc_metric(y_true[:, target], y_pred[:, target])
            if metric == 'ccc_norm':
                z = normalize_predictions(y_pred[:, target], mean_tr[target], std_tr[target])
                print('first ccc-normalized values:' + str(z[0:6]))
                _, val = ccc_metric(y_true[:, target], z)
            all_vals[metric].append(val)
            mystr += " " + metric + ":" + "{0:.3f}".format(val)
        myprint(mystr)
    return all_vals, mean_tr, std_tr


def avg_accuracy(string, accs):
    for target in range(len(targetsToPredict)):
        mystr = string + " " + target_names[target] + ": "
        for metric in metrics:
            s = 0
            for i_fold in range(nFold):
                if len(accs[i_fold]) == 0:
                    continue
                s += accs[i_fold][metric][target]
            s = s / nFold
            mystr += " AVG(" + metric + "):{0:.3f}".format(s)
        myprint(mystr)


def dump2arffnc(data, fname):
    with open(fname, 'wt') as file_pointer:
        arff.dump(data, file_pointer)
    call(['./fixarfffiles.sh', fname])
    call(['./arff2nc_noorder', fname, '2', str(len(targetsToPredict)), fname + '.nc'])


def save_data_to_nc(argss):
    data = argss[0]
    if len(argss) == 0:
        save_prefix = ''
    else:
        save_prefix = argss[1]
    myprint('Saving to nc...', Bcolors.OKGREEN)
    new_arff = {'relation': 'temp', 'attributes': [], 'data': []}
    targs = expand_labels(data).reshape(-1, len(targetsToPredict))
    for ii in range(len(data['data'])):
        t = [data['fname'][ii][0]] + [xx for xx in data['data'][ii, :]]
        t += [xx for xx in targs[ii, :]]
        new_arff['data'].append(t)
    new_arff['attributes'] = [('fname', 'STRING')] + attribute_names + [(xx, 'NUMERIC') for xx in target_names]
    fname = tmp_folder + save_prefix + '.arff'  # + ''.join(
    dump2arffnc(new_arff, fname)
    del new_arff
    return fname + '.nc'


def decode_currennt_result(csvname):
    with open(csvname, 'rt') as file_pointer:
        lines = file_pointer.readlines()
    result = [[] for _ in range(len(targetsToPredict))]
    for line in lines:
        cont = line.split(';')
        vals = [float(xx) for xx in cont[1:]]
        for iTarget in range(len(targetsToPredict)):
            result[iTarget] += vals[iTarget::len(targetsToPredict)]
    result = np.array(result)
    result = np.transpose(result)
    return result


def run_currennt(train, tr_fname, vs_fname, save_prefix=''):
    if train:
        myprint('Running Current for training...', Bcolors.OKGREEN)
        call(['./currennt',
              '--train', 'true',
              '--options_file', classifier['options_file'],
              '--network', classifier['network'],
              '--save_network', tmp_folder + save_prefix + 'trained_network.json',
              '--train_file', tr_fname,
              '--val_file', vs_fname])
    else:
        myprint('Running Current for testing...', Bcolors.OKGREEN)
        call(['./currennt',
              '--train', 'false',
              '--options_file', classifier['options_file'],
              '--network', tmp_folder + save_prefix + 'trained_network.json',
              '--ff_output_file', tmp_folder + save_prefix + 'result.csv',
              '--ff_input_file', vs_fname])
        return decode_currennt_result(tmp_folder + save_prefix + 'result.csv')


def expand_labels(data):
    seq = -1
    k = 0
    expanded = []
    new_val = 0
    for ii in range(data['seq'].shape[0]):
        if seq != data['seq'][ii]:
            new_val = data['target'][k]
            k += 1
            seq = data['seq'][ii]
        expanded.append(new_val)
    return np.array(expanded).ravel()


def compress_labels(labels, data):
    seq = data['seq'][0]
    new_lab = []
    v = []
    for ii in range(data['seq'].shape[0]):
        if seq != data['seq'][ii][0]:
            new_lab.append(np.mean(np.array(v), axis=0))
            v = []
            seq = data['seq'][ii]
        v.append(labels[i, :])
    new_lab.append(np.mean(np.array(v), axis=0))
    return np.array(new_lab)


def classify(train, dev, test, save_prefix=''):
    myprint('Classifying...', Bcolors.OKGREEN)
    pool = Pool(processes=3)
    argss = [[train, save_prefix + 'tr_'],
             [dev, save_prefix + 'dv_'],
             [test, save_prefix + 'ts_']]
    result = pool.map(save_data_to_nc, argss)
    pool.close()
    tr_fname = result[0]
    dv_fname = result[1]
    ts_fname = result[2]
    run_currennt(True, tr_fname, dv_fname, save_prefix)
    train_pred = run_currennt(False, '', tr_fname, save_prefix=save_prefix)
    dev_pred = run_currennt(False, '', dv_fname, save_prefix=save_prefix)
    test_pred = run_currennt(False, '', ts_fname, save_prefix=save_prefix)
    train_pred = compress_labels(train_pred, train)
    dev_pred = compress_labels(dev_pred, dev)
    test_pred = compress_labels(test_pred, test)
    return train_pred, dev_pred, test_pred


data_folds, target_names, attribute_names = load_data_into_folds()

nFold = len(data_folds)
acc_train = [[] for _ in range(nFold)]
acc_dev = [[] for _ in range(nFold)]
acc_test = [[] for _ in range(nFold)]
all_test_pred = [[] for _ in range(nFold)]
final_max_corr = []
final_test = []
for i in range(nFold):
    acc_train[i] = [[] for _ in range(nFold)]
    acc_dev[i] = [[] for _ in range(nFold)]
    acc_test[i] = [[] for _ in range(nFold)]
    all_test_pred[i] = [[] for _ in range(nFold)]

if test_on_last_fold:
    loop_over = range(nFold - 1, nFold)
else:
    loop_over = range(nFold)
for iFold in loop_over:  # only test
    myprint('Fold:' + str(iFold), Bcolors.OKGREEN)
    for jFold in range(nFold):  # dev
        if jFold == iFold:
            continue
        gc.collect()
        tr_data, dv_data, ts_data = prepare_data(data_folds, iFold, jFold)

        tr_data, args = normalize_data(tr_data, None)
        dv_data, _ = normalize_data(dv_data, args)
        ts_data, _ = normalize_data(ts_data, args)
        tr_pred, dv_pred, ts_pred = classify(tr_data, dv_data, ts_data, save_prefix=feat_set + '_' + str(iFold) + '_')

        tr_target = tr_data['target']
        dv_target = dv_data['target']
        ts_target = ts_data['target']
        acc_train[iFold][jFold], meant, stdt = calculate_accuracy("testFold:" + str(iFold) + " devFold:" + str(jFold) +
                                                                  " Train:", tr_target, tr_pred)
        acc_dev[iFold][jFold], _, _ = calculate_accuracy(
            "testFold:" + str(iFold) + " devFold:" + str(jFold) + " Dev:  ", dv_target, dv_pred, mean_tr=meant,
            std_tr=stdt)
        acc_test[iFold][jFold], _, _ = calculate_accuracy("testFold:" + str(iFold) + " devFold:" + str(jFold)
                                                          + " Test: ", ts_target, ts_pred,
                                                          mean_tr=meant, std_tr=stdt)

        for itarget in range(len(targetsToPredict)):
            ts_pred[:, itarget] = normalize_predictions(ts_pred[:, itarget], meant[itarget], stdt[itarget])
        all_test_pred[iFold][jFold] = ts_pred
    myprint('Average per fold accuracy...', Bcolors.OKGREEN)
    avg_accuracy("Train:", acc_train[iFold])
    avg_accuracy("Dev:  ", acc_dev[iFold])
    avg_accuracy("Test: ", acc_test[iFold])

    r = np.arange(0, nFold)
    r = np.delete(r, iFold)
    test_accuracies = [[] for _ in range(len(targetsToPredict))]
    mean_test = [[] for _ in range(len(targetsToPredict))]
    for itarget in range(len(targetsToPredict)):
        test_accuracies[itarget] = all_test_pred[iFold][r[0]]
        for i in range(1, len(r)):
            test_accuracies[itarget] = np.hstack((test_accuracies[itarget], all_test_pred[iFold][r[i]]))
        mean_test[itarget] = np.mean(test_accuracies[itarget], axis=1)
    mean_test = np.transpose(np.array(mean_test))
    res, _, _ = calculate_accuracy("Mean:testFold:" + str(iFold) + " Test: ", ts_target, mean_test, mean_tr=meant,
                                   std_tr=stdt)
    final_test.append(res)

if save_last_fold_result:
    with open(save_result_file, 'wt') as fp:
        x = [target_names[x] for x in targetsToPredict]
        x = str(x).replace(']', '').replace('[', '').replace('\'', '').replace(' ', '')
        fp.write('video,utterance,' + x + '\n')
        sorted_compressed_fnames = []
        prev = ''
        for i in range(len(ts_data['data'])):
            new = ts_data['fname'][i][0].split('.')[1]
            if prev != new:
                prev = new
                sorted_compressed_fnames.append(new)
        video_test = np.array([x.split('/')[0] for x in sorted_compressed_fnames])
        utt_test = np.array([x.split('/')[1] + '.mp4' for x in sorted_compressed_fnames])

        for ind, row in orig_test_order.iterrows():
            inds = np.nonzero(np.logical_and(row['video'] == video_test, row['utterance'] == utt_test))[0]
            if len(inds) != 0:  # repeating last value
                me = mean_test[inds[0], :]
                for itarget in range(len(targetsToPredict)):
                    if targetNormalize[itarget]:
                        me[itarget] = (me[itarget] + 1) / 2
                x = [str(x) for x in me]
                x = str(x).replace(']', '').replace('[', '').replace('\'', '').replace(' ', '')
            fp.write(row['video'] + ',' + row['utterance'] + ',' + x + '\n')
