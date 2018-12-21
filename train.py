import numpy as np
import time
import os
import shutil
import glob

from config import *
from model import GAReader
from utils import Helpers, DataPreprocessor, MiniBatchLoader
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.utils import to_categorical

class TensorBoardCustom(TensorBoard):

    def __init__(self, log_dir, words, **kwargs):
        super(TensorBoardCustom, self).__init__(log_dir, **kwargs)
        self.mapping_string = tf.constant(words)
        self.l_docin = tf.Variable([0., 0.])

    def _add_doc_summary(self):
        with tf.name_scope('summaries'):
            table = tf.contrib.lookup.index_to_string_table_from_tensor(
                    self.mapping_string, default_value="UNKNOWN")
            words = table.lookup(tf.cast(self.l_docin, tf.int64))
            text = tf.reduce_join(words, 0, separator=' ')
            tf.summary.text('text', text)

    def on_batch_end(self, batch, logs={}):
#        print(logs)
#        print(self.model.variables)
#        tf.assign(self.l_docin, self.model.inputs[0][0], validate_shape=False)
#        self._add_doc_summary()
        writer = tf.summary.FileWriter(self.log_dir)
        lr = tf.summary.scalar('learning_rate', self.model.optimizer.lr)
        summary = tf.summary.merge_all() 
        writer.add_summary(summary.numpy(), batch)
        writer.close()
        super(TensorBoardCustom, self).on_batch_end(batch, logs)

def main(save_path, params):

    nhidden = params['nhidden']
    dropout = params['dropout']
    word2vec = params['word2vec']
    dataset = params['dataset']
    nlayers = params['nlayers']
    train_emb = params['train_emb']
    char_dim = params['char_dim']
    use_feat = params['use_feat']
    gating_fn = params['gating_fn']
    out = 'out'

    # save settings
    shutil.copyfile('config.py','%s/config.py'%save_path)

    use_chars = char_dim>0
    dp = DataPreprocessor.DataPreprocessor()
    data = dp.preprocess(dataset, no_training_set=False, use_chars=use_chars)
    word_dictionary = data.dictionary[0]
    the_index = word_dictionary['the']
    print('the index : {}'.format(word_dictionary['the']))

    idx_to_word = dict([(v, k) for (k, v) in word_dictionary.iteritems()])
    words = [idx_to_word[i] for i in sorted(idx_to_word.keys())]

    print("building minibatch loaders ...")
    batch_loader_train = MiniBatchLoader.MiniBatchLoader(data.training, BATCH_SIZE, 
            sample=1.0)
    batch_loader_val = MiniBatchLoader.MiniBatchLoader(data.validation, BATCH_SIZE)

    print("building network ...")
    W_init, embed_dim, = Helpers.load_word2vec_embeddings(data.dictionary[0], word2vec)
    print('the embedding : {}'.format(W_init[the_index]))
    print(W_init[0:5])

    print("running GAReader ...")

    m = GAReader.Model(nlayers, data.vocab_size, data.num_chars, W_init, 
            nhidden, embed_dim, dropout, train_emb, 
            char_dim, use_feat, gating_fn, words).build_network()
    m.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, clipnorm=GRAD_CLIP),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
    saver = tf.train.Saver(m.weights, max_to_keep=1)
    #tf.enable_eager_execution(config=tf.ConfigProto(allow_soft_placement = True))
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess:
            K.set_session(sess)
                #with tf.device('/gpu:0:'):
            ckpts = glob.glob('output/weights*.hd5')
            if len(ckpts) > 0:
                ckpts = sorted(ckpts)
                print('loading model from checkpoint : {}'.format(ckpts[-1]))
                saver.restore(sess, ckpts[-1])
                print(m.get_weights()[0])
            tensorboard = TensorBoardCustom(log_dir="logs", words=words)
            modelcheckpoint = tf.keras.callbacks.ModelCheckpoint('output/weights.{epoch:02d}-{val_loss:.2f}.hdf5')
            writer = tf.summary.FileWriter("logs")
            
            def schedule(epoch, lr):
                
                if epoch >= 3:
                    return lr * 0.5
                else:
                    return lr
            lrate = LearningRateScheduler(schedule, verbose=1)

            for epoch in xrange(NUM_EPOCHS):
                for (inputs, a) in batch_loader_train:
                    [dw, qw, m_dw, m_qw, c, m_c, cl] = inputs
                    m = GAReader.Model(max_doc_len, max_qry_len, nlayers, data.vocab_size, data.num_chars, W_init, 
                            nhidden, embed_dim, dropout, train_emb, 
                            char_dim, use_feat, gating_fn, words).build_network()
                    m.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, clipnorm=GRAD_CLIP),
                              loss=tf.keras.losses.categorical_crossentropy,
                              metrics=[tf.keras.metrics.categorical_accuracy])
                    #print(dw.shape)
                    #print('dw : {}'.format(dw))
                    #print('qw : {}'.format(qw))
                    #print('m_dw : {}'.format(m_dw))
                    #print('m_qw : {}'.format(m_qw))
                    #print('c : {}'.format(c))
                    #print([idx_to_word[i] for i in dw[0, :, 0].tolist()])
                    train_summary = m.train_on_batch(inputs, to_categorical(a, batch_loader_train.max_num_cand))
                    print(m.get_weights()[0])
                    print('epoch: {}, train loss: {}, train acc: {}'.format(epoch, train_summary[0], train_summary[1]))
                    lr = tf.summary.scalar('learning_rate', LEARNING_RATE)
                    summary = tf.summary.merge_all() 
                    s = sess.run(summary)
                    writer.add_summary(s)
                saver.save(sess, 'output/weights.epoch_{:02}-val_loss_{:03.2f}.ckpt'.format(epoch, 0.0))
                #m.save_weights('output/weights.epoch:{:2}-val_loss:{:.2}.hdf5'.format(epoch, 0.0))
                writer.close()
    
