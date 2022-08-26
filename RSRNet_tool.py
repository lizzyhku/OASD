# -*- coding: utf-8 -*-

import numpy as np
from parameter import FLAGS
from data_utils import G

def refine(route, output):
    reoutput = [output[0]]
    count = 0
    for i in range(1, len(route)-1):
        if G.out_degree(route[i-1])==1 and G.in_degree(route[i])==1:
            reoutput.append(reoutput[-1])
            count += 1
        else:
            reoutput.append(output[i])
    reoutput.append(output[-1])
    return np.array(reoutput)

def train_rsr(model, label, text, signal):
    loss = .0
    signal_ = []
    for sig in signal:
        signal_.append(np.eye(model.num_signal, dtype=np.int)[sig])
    batched_data = {'texts': np.array(text), 'labels':np.array(label).reshape(-1), 'keep_prob':FLAGS.keep_prob, 'signal': np.array(signal_).reshape(-1,model.num_signal)}
    outputs = model.train_step(model.sess_rsr, batched_data)
    loss += outputs[0]
    return loss / len(label), batched_data 

def evaluate_rsr(model, label, text, signal):
    loss = .0
    signal_ = []
    for sig in signal:
        signal_.append(np.eye(model.num_signal, dtype=np.int)[sig])
    batched_data = {'texts': np.array(text), 'labels':np.array(label).reshape(-1), 'keep_prob':FLAGS.keep_prob, 'signal': np.array(signal_).reshape(-1,model.num_signal)}
    outputs = model.test_step(model.sess_rsr, batched_data)
    loss += outputs[0]
    return loss

def inference_rsr(model, text, signal):
    signal_ = []
    for sig in signal:
        signal_.append(np.eye(model.num_signal, dtype=np.int)[sig])
    hidden_states = np.zeros([len(text), FLAGS.hidden_units], dtype=np.float32)
    batched_data = {'texts': np.array(text), 'keep_prob':FLAGS.keep_prob, 'signal': np.array(signal_).reshape(-1,model.num_signal)}
    hidden_states = model.sess_rsr.run(model.outputs, {model.texts:batched_data['texts'], model.keep_prob:1.0, model.signal:batched_data['signal']})
    return hidden_states
