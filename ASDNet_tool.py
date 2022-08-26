# -*- coding: utf-8 -*-

import numpy as np
from parameter import FLAGS
from data_utils import is_decision

def similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def from_RSR_loss(model, batch_obs, text, SRN, batched_data):
    label_pred = evaluate_asd(model, batch_obs, text)
    input_feed = {SRN.texts: batched_data['texts'],
                SRN.labels: np.array(label_pred).reshape(-1),
                SRN.keep_prob: batched_data['keep_prob'],
                SRN.length_se_t: np.full(1, len(batched_data['labels']), dtype=np.int32),
                SRN.signal: batched_data['signal']}
    return 1/(1+SRN.sess_rsr.run([SRN.loss], input_feed)[0]/len(label_pred))

def determine(model, batch_obs, labels, text, SRN, batched_data):
    reward = [.0]
    model.labels = []
    model.local_reward = 0.0
    count_local_reward = 0
    flag = []
    for idx, obs in enumerate(batch_obs):
        if idx == 0 or idx == len(batch_obs)-1:
            is_dec, tag = False, 0
        else:
            is_dec, tag = is_decision(text[idx-1], text[idx], model.labels[-1])
        if is_dec:
            flag.append(idx)
            action = labels[idx]
            model.local_reward += FLAGS.continuity * (int(model.labels[-1] == labels[idx]) - int(model.labels[-1] != labels[idx])) * similarity(batch_obs[idx-1], obs)
            count_local_reward += 1
            obs_label = np.eye(FLAGS.labels, dtype=np.int)[model.labels[-1]]
            model.store_transition(obs, obs_label, action, 0)
        model.labels.append(labels[idx])
    if count_local_reward != 0:
        model.local_reward /= count_local_reward
    global_reward = from_RSR_loss(model, batch_obs, text, SRN, batched_data)
    reward[0] = global_reward + model.local_reward
    return reward

def pretrain_asd(model, observations, labels, text, SRN, batched_data):
    loss = .0    
    reward = determine(model, observations, labels, text, SRN, batched_data)
    if model.ep_as[0] != []:
        loss += model.prelearn(model.sess_asd, reward, FLAGS.keep_prob)
    return loss / len(observations)

def explore(model, batch_obs, labels, text, SRN, batched_data, delay=5):
    reward = [.0] * FLAGS.sample_round
    delay = FLAGS.delay
    check_start_idx = []
    end_start_idx = None
    for n_round in range(FLAGS.sample_round):
        model.labels = []
        model.local_reward = 0.0
        count_local_reward = 0
        flag = []
        for idx, obs in enumerate(batch_obs):
            if len(check_start_idx) != 0:
                delay -= 1
                if model.labels[idx-1] == 1:
                    end_start_idx = idx
            #reset
            if delay == 0: 
                if end_start_idx is not None:
                    model.labels[check_start_idx[0]:end_start_idx] = [1] * (end_start_idx - check_start_idx[0])  
                    del(check_start_idx[0])
                    end_start_idx = None
                if len(check_start_idx) != 0:
                    delay = FLAGS.delay - (idx - check_start_idx[0] - 1)
                else:
                    delay = FLAGS.delay        
            if idx == 0 or idx == len(batch_obs)-1:
                is_dec, tag = False, 0
            else:
                is_dec, tag = is_decision(text[idx-1], text[idx], model.labels[-1])
            if is_dec:
                flag.append(idx)
                obs_label = np.eye(FLAGS.labels, dtype=np.int)[model.labels[-1]]
                action = model.choose_action(model.sess_asd, obs, obs_label, True)
                model.labels.append(action)
                model.local_reward += FLAGS.continuity * (int(model.labels[-1] == model.labels[-2]) - int(model.labels[-1] != model.labels[-2])) * similarity(batch_obs[idx-1], obs)
                count_local_reward += 1
                if model.labels[-1] == 0 and model.labels[-2]==1:
                    check_start_idx.append(idx)
            else:
                model.labels.append(tag)    
            if is_dec:
                model.store_transition(obs, obs_label, action, n_round)
        if count_local_reward != 0:
            model.local_reward /= count_local_reward
        global_reward = from_RSR_loss(model, batch_obs, text, SRN, batched_data)
        reward[n_round] = global_reward + model.local_reward
    return reward
        
def train_asd(model, observations, labels, text, SRN, batched_data):
    reward = explore(model, observations, labels, text, SRN, batched_data)
    loss = model.learn(model.sess_asd, reward, FLAGS.keep_prob)    
    return loss

def evaluate_asd(model, observations, text, delay=5):
    model.labels = []
    delay = FLAGS.delay
    check_start_idx = []
    end_start_idx = None
    for idx, obs in enumerate(observations):
        if len(check_start_idx) != 0:
            delay -= 1
            if model.labels[idx-1] == 1:
                end_start_idx = idx
        #reset
        if delay == 0:
            if end_start_idx is not None:
                model.labels[check_start_idx[0]:end_start_idx] = [1] * (end_start_idx - check_start_idx[0])  
                del(check_start_idx[0])
                end_start_idx = None
            if len(check_start_idx) != 0:
                delay = FLAGS.delay - (idx - check_start_idx[0] - 1)
            else:
                delay = FLAGS.delay
            
        if idx == 0 or idx == len(observations)-1:
            is_dec, tag = False, 0
        else:
            is_dec, tag = is_decision(text[idx-1], text[idx], model.labels[-1])
        if is_dec:
            obs_label = np.eye(FLAGS.labels, dtype=np.int)[model.labels[-1]]
            action = model.choose_action(model.sess_asd, obs, obs_label, False)
            model.labels.append(action)
            if model.labels[-1] == 0 and model.labels[-2]==1:
                check_start_idx.append(idx)
        else:
            model.labels.append(tag)
    return model.labels