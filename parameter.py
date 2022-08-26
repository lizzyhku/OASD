# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

traj_path = './data/'

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("symbols", 6113, "vocabulary size.") #max(w2v_toast.vocab.keys())
tf.app.flags.DEFINE_integer("labels", 2, "Number of topic labels.")
tf.app.flags.DEFINE_integer("epoch_pre", 1, "Number of epoch on pretrain.")
tf.app.flags.DEFINE_integer("epoch_max", 5, "Maximum of epoch in iterative training.")
tf.app.flags.DEFINE_integer("embed_units", 128, "Size of word embedding.")# w2v_toast.vector_size
tf.app.flags.DEFINE_integer("hidden_units", 128, "Size of hidden layer.")
tf.app.flags.DEFINE_integer("sample_round", 1, "Sample round in RL.")
tf.app.flags.DEFINE_float("continuity", 1.0, "Coefficient of continuity reward.")
tf.app.flags.DEFINE_float("learning_rate_rsr", 0.01, "RSRNet Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_asd", 0.001, "ASDNet Learning rate.")
tf.app.flags.DEFINE_float("keep_prob", 0.0, "Fraction of the input units to drop.")
tf.app.flags.DEFINE_string("train_dir_rsr", traj_path+"train_rsr", "RSRNet Training directory.")
tf.app.flags.DEFINE_string("train_dir_asd", traj_path+"train_asd", "ASDNet Training directory.")
tf.app.flags.DEFINE_string("rsr_model", traj_path+"train_rsr/trained_model", "RSRNet load model.")
tf.app.flags.DEFINE_string("asd_model", traj_path+"train_asd/trained_model", "ASDNet load model.")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size for train.")
tf.app.flags.DEFINE_float("threshold", 0.5, "the prior proba for noise labelling.")
tf.app.flags.DEFINE_float("reward_decay", 0.99, "reward discount rate.")
tf.app.flags.DEFINE_integer("pre", 100, "Number of data on pretrain.")
tf.app.flags.DEFINE_integer("delay", 8, "delay parameter.")