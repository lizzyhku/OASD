# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
tf.reset_default_graph()

tf.set_random_seed(0)

class ASDNetwork(object):
    def __init__(self,
                 num_actions,
                 num_features,
                 sample_round,
                 learning_rate=0.001,
                 max_gradient_norm=5.0,
                 reward_decay=0.95):
        self.n_actions = num_actions
        self.n_features = num_features
        self.sample_round = sample_round
        self.gamma = reward_decay
        self.ep_obs, self.ep_obs_label, self.ep_as = [[] for _ in range(self.sample_round)], [[] for _ in range(self.sample_round)], [[] for _ in range(self.sample_round)]
        self.labels = []
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features*2], name="observations")
            self.tf_idx = tf.placeholder(tf.float32, [None, self.n_actions], name="indices")
            self.tf_acts = tf.placeholder(tf.int32, [None], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None], name="actions_value")
            self.keep_prob = tf.placeholder(tf.float32)
        
        with tf.name_scope('state'):
            topic_vec = tf.layers.dense(self.tf_idx, self.n_features, name='label_vec')
            all_obs = tf.concat([self.tf_obs, topic_vec], 1)
            self.all_act = tf.layers.dense(all_obs, self.n_actions, name='fc') 
        
        with tf.name_scope('policy'):
            self.all_act_prob = tf.nn.softmax(self.all_act, name='act_prob') 
            
        with tf.name_scope('loss'):
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
        
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        self.sess_asd = tf.Session()
        self.saver = tf.train.Saver()
        self.sess_asd.run(tf.global_variables_initializer())
        
    def save(self, checkpoint):
        self.saver.save(self.sess_asd, checkpoint, global_step=self.global_step)
    
    def load(self, checkpoint, reader_name):
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('training from last checkpoint', checkpoint)
            self.saver.restore(self.sess_asd, ckpt.model_checkpoint_path)                
        #reader = pywrap_tensorflow.NewCheckpointReader(reader_name)
        #Print tensor name and values
        #var_to_shape_map = reader.get_variable_to_shape_map()
        #for key in var_to_shape_map:
            #print("tensor_name: ", key)
        #    print(reader.get_tensor(key).shape)
        #self.bias_1 = reader.get_tensor('label_vec/bias')
        #self.kernel_1 = reader.get_tensor('label_vec/kernel')
        #self.bias_2 = reader.get_tensor('fc/bias')
        #self.kernel_2 = reader.get_tensor('fc/kernel')              
    
    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
    
    def choose_action(self, session, observation, ref_label, is_train):
        prob_weights = session.run(self.all_act_prob, feed_dict={self.tf_idx: [ref_label], self.keep_prob: 1.0,
                self.tf_obs: observation[np.newaxis, :]})   # shape=[1, n_features]    
        prob_weights = prob_weights.ravel()
        
        # select action w.r.t the actions prob
        action = np.random.choice(len(prob_weights), p=prob_weights) if is_train else np.argmax(prob_weights)
        return action
    
    def store_transition(self, s, s_label, a, n_round):
        self.ep_obs[n_round].append(s)
        self.ep_as[n_round].append(a)
        self.ep_obs_label[n_round].append(s_label)
    
    def learn(self, session, reward, keep_prob):
        loss = .0
        for n_round in range(self.sample_round):
            if self.ep_as[n_round] == []:
                continue
            # discount episode rewards
            seq_len = len(self.ep_as[n_round])
            discounted_ep_rs = np.array([reward[n_round]] * seq_len).T
            outputs = session.run([self.loss, self.train_op], feed_dict={
                    self.tf_obs: np.array(self.ep_obs[n_round]),    # shape=[seq_len, 2*n_features]
                    self.tf_idx: np.array(self.ep_obs_label[n_round]),
                    self.tf_acts: np.array(self.ep_as[n_round]),    # shape=[seq_len]
                    self.tf_vt: discounted_ep_rs,                   # shape=[seq_len]
                    self.keep_prob: keep_prob})
            loss += outputs[0]
        self.clean()
        return loss / self.sample_round
        
    def prelearn(self, session, reward, keep_prob):
        seq_len = len(self.ep_as[0])
        discounted_ep_rs = np.array([reward[0]] * seq_len).T
        outputs = session.run([self.loss, self.train_op], feed_dict={
                self.tf_obs: np.array(self.ep_obs[0]),
                self.tf_idx: np.array(self.ep_obs_label[0]),
                self.tf_acts: np.array(self.ep_as[0]),
                self.tf_vt: discounted_ep_rs,
                self.keep_prob: keep_prob})
        loss = outputs[0]
        self.clean()
        return loss
    
    def clean(self):
        self.ep_obs, self.ep_obs_label, self.ep_as = [[] for _ in range(self.sample_round)], [[] for _ in range(self.sample_round)], [[] for _ in range(self.sample_round)]
        