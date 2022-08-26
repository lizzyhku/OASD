# -*- coding: utf-8 -*-

import tensorflow as tf
tf.reset_default_graph()
import numpy as np
tf.set_random_seed(0)

class RSRNetwork(object):
    def __init__(self,
                 num_signal,
                 num_symbols,
                 num_embed_units,
                 num_units,
                 embed,
                 learning_rate=0.001,
                 max_gradient_norm=5.0
                 ):
        self.texts = tf.placeholder(tf.int32, [None]) # shape: sentence
        self.labels = tf.placeholder(tf.int32, [None])      # shape: sentence
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.num_signal = num_signal
        
        self.length_se_t = tf.placeholder(tf.int32, [None])
        self.signal = tf.placeholder(tf.float32, [None, self.num_signal])  # related to previous action
        
        # build the embedding table (index to vector)
        self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed, trainable=True)
        self.embed_inputs = tf.nn.embedding_lookup(self.embed, self.texts)   # shape: sentence*num_embed_units
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
        self.embed_inputs = tf.expand_dims(self.embed_inputs, 0)    # shape: 1*sentence*num_embed_units
        outputs, last_states = tf.nn.dynamic_rnn(cell=cell, inputs=self.embed_inputs, dtype=tf.float32, time_major=False) 
        outputs_ = outputs[0]   # shape: sentence*num_units
        signal_vec = tf.layers.dense(self.signal, num_units, name='signal_vec')
        self.outputs = tf.concat([outputs_, signal_vec], 1)
        self.logits = tf.layers.dense(self.outputs, 2) 
        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits), name='loss')
        mean_loss = self.loss / tf.cast(tf.shape(self.labels)[0], dtype=tf.float32)
                
        self.params = tf.trainable_variables()
        # calculate the gradient of parameters
        
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(mean_loss, self.params)
        
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), global_step=self.global_step)
        
        self.sess_rsr = tf.Session()
        self.saver = tf.train.Saver()
        self.sess_rsr.run(tf.global_variables_initializer())
        
    def save(self, checkpoint):
        self.saver.save(self.sess_rsr, checkpoint, global_step=self.global_step)
    
    def load(self, checkpoint):
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('training from last checkpoint', checkpoint)
            self.saver.restore(self.sess_rsr, ckpt.model_checkpoint_path)     
        
    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
            
    def train_step(self, session, data):
        input_feed = {self.texts: data['texts'],
                self.labels: data['labels'],
                self.keep_prob: data['keep_prob'],
                self.length_se_t: np.full(1, len(data['labels']), dtype=np.int32),
                self.signal: data['signal']}
        output_feed = [self.loss, self.logits, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)
    
    def test_step(self, session, data):
        input_feed = {self.texts: data['texts'],
                self.labels: data['labels'],
                self.keep_prob: data['keep_prob'],
                self.length_se_t: np.full(1, len(data['labels']), dtype=np.int32),
                self.signal: data['signal']}
        output_feed = [self.loss, self.logits] 
        return session.run(output_feed, input_feed)
 