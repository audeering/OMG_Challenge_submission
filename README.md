----------VIDEO ANALYSIS: valence: Submission #1
Code structure is as follows:

Multi_File_Extraction.py is used to extract features per video
Combine_Features_Per_Video.py is then used to create a single csv file from all the features
Batched_Stateless_Prediction_Video_Fold_Ensemble.py is used for training models and making predictions on test set


Dependecies: 
- MTCNN library: https://github.com/kpzhang93/MTCNN_face_detection_alignment 
- caffe
- tensorflow
- keras
- numpy
- pandas
- sklearn

Usage:
First run Multi_File_Extraction.py to extract video features.
Set appropriate variables as follows:
OMG_Data_Description_root: folder where omg_TrainVideos.csv etc. description files are placed
database: root folder where videos are downloaded, separated into three subfolders 'dev_data', 'train_data', 'test_data'
results: root folder to place results in
caffe_root: root folder for caffe installation
caffe_model_path: path to VGGFace and MTCNN models placed in appropriate folders (see script for info)

----------AUDIO ANALYSIS:
compile netcdf:
- download netcdf 4.6.1: ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.6.1.tar.gz
- Change the include path to the netcdf.h in the nc-standardize.cpp
- Compile the nc-standardize:
g++ nc-standardize.cpp -lnetcdf -onc-standardize

build currennt (NN classifier) and fix the path to the currennt in the analyser.py:
- follow the instructions in the currennt/README file

converting to wav:
- run convert2wav.sh on the three train/dev/test folders

Feature Extraction:
- Change the 'fold' ('train','dev','test') and 'data_folder' variables 
  in the FX.py and run it.

Classification:
inside the analyser.py
Arousal: Submission #1: With Random permutation of the samples and folding:
- shuffle_training_data = True

Arousal: Submission #2: spkear-independent folds
- shuffle_training_data = False

Arousal: Submissoin #3:
- averagin over submission #2 and #3

Valence: Submission #2: 
- shuffle_training_data = True
- targetsToPredict = [0, 1]
- targetNormalize = [True, False]

and run the analyser_clean.py

Valence: Submission #3:
Average of audio (submission #2) and video valence results

