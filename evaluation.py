# -*- coding: utf-8 -*-

import time
from parameter import FLAGS
from RSRNet import RSRNetwork
from RSRNet_tool import inference_rsr
from ASDNet import ASDNetwork
from ASDNet_tool import evaluate_asd
import data_utils as F
import pickle

globaltime = str(time.time())
traj_path = './data/'
num_signal = 2

[LABEL_TEST, TEXT_TEST, SENTENCE_LEN_TEST, NORMAL_SIGNAL_TEST] = pickle.load(open(traj_path+'TESTSET', 'rb'), encoding='bytes')

embed = F.build_embed(FLAGS.symbols, FLAGS.embed_units)
RSR = RSRNetwork(num_signal,
              FLAGS.symbols,
              FLAGS.embed_units,
              FLAGS.hidden_units,
              embed,
              FLAGS.learning_rate_rsr)    
    
RSR.load(FLAGS.rsr_model)        

ASD = ASDNetwork(FLAGS.labels,
                       FLAGS.embed_units,
                       FLAGS.sample_round,
                       FLAGS.learning_rate_asd,
                       FLAGS.reward_decay)    

ASD.load(FLAGS.asd_model, FLAGS.asd_model+'/checkpoint-0')

def check(name):
    anoma_road = []
    total_road = []
    Label_Test, Label_Pred = [], []
    for label_test, text_test, normal_signal_test in zip(LABEL_TEST, TEXT_TEST, NORMAL_SIGNAL_TEST): #            
        obs_test = inference_rsr(RSR, text_test, normal_signal_test)        
        label_pred = evaluate_asd(ASD, obs_test, text_test)
        
        anoma_road.append(sum(label_pred[1:-1]))
        total_road.append(len(label_pred[1:-1]))

        Label_Test.append(label_test)
        Label_Pred.append(label_pred)

    labelling = F.my_fscore_whole_determine(Label_Test, Label_Pred)
    print('labelling', labelling)
    return Label_Test, Label_Pred

Label_Test, Label_Pred = check('my_model')
