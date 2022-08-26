# -*- coding: utf-8 -*-

import time
from parameter import FLAGS
from RSRNet import RSRNetwork
from RSRNet_tool import train_rsr, inference_rsr
from ASDNet import ASDNetwork
from ASDNet_tool import pretrain_asd, train_asd, evaluate_asd
import data_utils as F
import pickle

globaltime = str(time.time())

traj_path = './data/'

LABEL_TRAIN, TEXT_TRAIN, SENTENCE_LEN_TRAIN, SD_SLOT_TRAIN, NORMAL_SIGNAL_TRAIN = F.load_data(FLAGS.threshold, FLAGS.batch_size*100, consider=0.4)

[LABEL_TRAIN_PRE, TEXT_TRAIN_PRE, SENTENCE_LEN_TRAIN_PRE , NORMAL_SIGNAL_TRAIN_PRE], \
[LABEL_TEST, TEXT_TEST, SENTENCE_LEN_TEST, NORMAL_SIGNAL_TEST] = F.load_data_groundtruth_sd(consider=0.4)

LABEL_TEST, TEXT_TEST, SENTENCE_LEN_TEST, NORMAL_SIGNAL_TEST = LABEL_TEST+LABEL_TRAIN_PRE, TEXT_TEST+TEXT_TRAIN_PRE, SENTENCE_LEN_TEST+SENTENCE_LEN_TRAIN_PRE, NORMAL_SIGNAL_TEST+NORMAL_SIGNAL_TRAIN_PRE
pickle.dump([LABEL_TEST, TEXT_TEST, SENTENCE_LEN_TEST, NORMAL_SIGNAL_TEST], open(traj_path+'TESTSET', 'wb'), protocol=2)

LABEL_TRAIN_PRE, TEXT_TRAIN_PRE, NORMAL_SIGNAL_TRAIN_PRE = LABEL_TRAIN_PRE[200:300], TEXT_TRAIN_PRE[200:300], NORMAL_SIGNAL_TRAIN_PRE[200:300] #development set

best_labelling = 0
num_signal = 2
embed = F.build_embed(FLAGS.symbols, FLAGS.embed_units)

RSR = RSRNetwork(num_signal,
              FLAGS.symbols,
              FLAGS.embed_units,
              FLAGS.hidden_units,
              embed,
              FLAGS.learning_rate_rsr)    
#RSR.load(FLAGS.rsr_model)        

ASD = ASDNetwork(FLAGS.labels,
                       FLAGS.embed_units,
                       FLAGS.sample_round,
                       FLAGS.learning_rate_asd,
                       FLAGS.reward_decay)    
#ASD.load(FLAGS.asd_model)

def check(name):
    anoma_road = []
    total_road = []
    Label_Test, Label_Pred = [], []
    for label_test, text_test, normal_signal_test in zip(LABEL_TRAIN_PRE, TEXT_TRAIN_PRE, NORMAL_SIGNAL_TRAIN_PRE):
        obs_test = inference_rsr(RSR, text_test, normal_signal_test)        
        label_pred = evaluate_asd(ASD, obs_test, text_test)
        anoma_road.append(sum(label_pred[1:-1]))
        total_road.append(len(label_pred[1:-1]))

        Label_Test.append(label_test)
        Label_Pred.append(label_pred)

    labelling = F.my_fscore_whole_determine(Label_Test, Label_Pred)
    print('labelling', labelling)
    global best_labelling

    if labelling > best_labelling:
        print("==============================================================")
        best_labelling = labelling
        RSR.save('%s/%s_%s_res_%s/checkpoint' % (FLAGS.train_dir_rsr, globaltime, name, best_labelling))
        ASD.save('%s/%s_%s_res_%s/checkpoint' % (FLAGS.train_dir_asd, globaltime, name, best_labelling))
        print('best labelling-fscore {} with {} / {}'.format(best_labelling, sum(anoma_road), sum(total_road)))
        print("==============================================================")
    return Label_Test, Label_Pred

##################################################################
start = time.time()
print("Start pre-training ...")
pre_train_count = 0
count, check_valid = 0, 10

while True:
    if pre_train_count == 1:
        break
    print('pre-training with epoch', pre_train_count)
    for label_train, text_train, normal_signal in zip(LABEL_TRAIN[:200], TEXT_TRAIN[:200], NORMAL_SIGNAL_TRAIN[:200]): # 
        count += 1
        for _ in range(FLAGS.epoch_pre):
            loss, batched_data = train_rsr(RSR, label_train, text_train, normal_signal)
            
        obs_train = inference_rsr(RSR, text_train, normal_signal)
        for _ in range(FLAGS.epoch_pre):
            loss = pretrain_asd(ASD, obs_train, label_train, text_train, RSR, batched_data)
        if count % check_valid == 0:
            print('in pre-train', count)
            check('pre-train')    
        ASD.clean()
    pre_train_count += 1

##################################################################
print("Start training ...")
count = 0
train_count, train_t = 0, 1
save_record = []

while True:
    if train_count == train_t:
        break    
    train_count += 1
    for label_train, text_train, normal_signal_train in zip(LABEL_TRAIN, TEXT_TRAIN, NORMAL_SIGNAL_TRAIN):           
        if count % 1000 == 0 or count == len(LABEL_TRAIN)-1:
            print('process time', count, time.time()-start, best_labelling)
            save_record.append([time.time()-start, best_labelling])
        count += 1   
        for _ in range(FLAGS.epoch_max):
            if _ > 0:
                loss, batched_data = train_rsr(RSR, label_train, text_train, normal_signal_train)
            obs_train = inference_rsr(RSR, text_train, normal_signal_train)                  
            if _ > 0:
                loss = train_asd(ASD, obs_train, label_train, text_train, RSR, batched_data)   
            label_train = evaluate_asd(ASD, obs_train, text_train)
        ASD.clean()
        if count % 10 == 0:
            check('train')    
print('best labelling-fscore {}'.format(best_labelling))
