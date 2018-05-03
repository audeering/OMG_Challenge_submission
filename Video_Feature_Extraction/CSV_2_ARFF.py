'''
This code is used to turn video features csv files into the arff format.

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

def CSV_2_ARFF(filename_in, filename_out, network_name, layer_name, relation_name = '@relation Video_Features'):
    with open(filename_out, 'w') as fout:
        fout.write(relation_name + '\n')
        fout.write('\n')
        with open(filename_in, 'r') as fin:
            l = fin.readline() # read first line where names of features are
            l = l.split(',')
            for feature in l:
                feature = feature.strip('\n')
                if feature == 'name':
                    fout.write('@attribute ' + feature + ' string\n')
                elif feature in ['valence', 'arousal', 'emotion', 'face']:
                    fout.write('@attribute ' + feature + ' numeric\n')
                else:
                    fout.write('@attribute ' + network_name + '_' + layer_name + '_' + feature + ' numeric\n')
                
            fout.write('\n')
            fout.write('@data\n')
            fout.write('\n')
            for l in fin:
                fout.write(l)
#                fout.write('\n')
                
if __name__ == '__main__':
    
    CSV_2_ARFF('../DNN_Results/DFL_f5_Dev_Combined.csv', 
               '../DNN_Results/DFL_f5_Dev_Combined.arff',
               'DFL', 'f5')    
    CSV_2_ARFF('../DNN_Results/DFL_f5_Train_Combined.csv', 
               '../DNN_Results/DFL_f5_Train_Combined.arff',
               'DFL', 'f5')  


    CSV_2_ARFF('../DNN_Results/SphereFace_f5_Dev_Combined.csv', 
               '../DNN_Results/SphereFace_f5_Dev_Combined.arff',
               'SphereFace', 'f5')    
    CSV_2_ARFF('../DNN_Results/SphereFace_f5_Train_Combined.csv', 
               '../DNN_Results/SphereFace_f5_Train_Combined.arff',
               'SphereFace', 'f5')      

    CSV_2_ARFF('../DNN_Results/VGGFace_f6_Dev_Combined.csv', 
               '../DNN_Results/VGGFace_f6_Dev_Combined.arff',
               'VGGFace', 'f6')    
    CSV_2_ARFF('../DNN_Results/VGGFace_f6_Train_Combined.csv', 
               '../DNN_Results/VGGFace_f6_Train_Combined.arff',
               'VGGFace', 'f6')  

    CSV_2_ARFF('../DNN_Results/VGGFace_f7_Dev_Combined.csv', 
               '../DNN_Results/VGGFace_f7_Dev_Combined.arff',
               'VGGFace', 'f7')    
    CSV_2_ARFF('../DNN_Results/VGGFace_f7_Train_Combined.csv', 
               '../DNN_Results/VGGFace_f7_Train_Combined.arff',
               'VGGFace', 'f7')      
    
    CSV_2_ARFF('../DNN_Results/VGGFace_f8_Dev_Combined.csv', 
               '../DNN_Results/VGGFace_f8_Dev_Combined.arff',
               'VGGFace', 'f8')    
    CSV_2_ARFF('../DNN_Results/VGGFace_f8_Train_Combined.csv', 
               '../DNN_Results/VGGFace_f8_Train_Combined.arff',
               'VGGFace', 'f8')
