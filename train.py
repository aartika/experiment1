import numpy as np
import time
import os
import shutil

from config import *
from model import GAReader
from utils import Helpers, DataPreprocessor, MiniBatchLoader
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.utils import to_categorical

class TensorBoardCustom(TensorBoard):

    def __init__(self, log_dir, sess, words, **kwargs):
        super(TensorBoardCustom, self).__init__(log_dir, **kwargs)
        self.sess = sess
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
        s = self.sess.run(summary)
        writer.add_summary(s, batch)
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
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess:
                K.set_session(sess)
                tf.train.create_global_step()
                #with tf.device('/gpu:0:'):
                m = GAReader.Model(nlayers, data.vocab_size, data.num_chars, W_init, 
                        nhidden, embed_dim, dropout, train_emb, 
                        char_dim, use_feat, gating_fn, words).build_network()
                m.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, clipnorm=GRAD_CLIP),
                          loss=tf.keras.losses.categorical_crossentropy,
                          metrics=[tf.keras.metrics.categorical_accuracy])
                tensorboard = TensorBoardCustom(log_dir="logs", sess=sess, words=words)
                writer = tf.summary.FileWriter("logs")
                
                def schedule(epoch, lr):
                    if epoch >= 3:
                        return lr * 0.5
                    else:
                        return lr
                lrate = LearningRateScheduler(schedule, verbose=1)

                #m.fit_generator(generator=batch_loader_train, steps_per_epoch=len(batch_loader_train.batch_pool), epochs=100, callbacks=[tensorboard, lrate])
                #validation_data=batch_loader_val, validation_steps=len(batch_loader_val.batch_pool))
                for (inputs, a) in batch_loader_train:
                    [dw, qw, m_dw, m_qw, c, m_c, cl] = inputs
                    print(dw.shape)
                    print('dw : {}'.format(dw))
                    print('qw : {}'.format(qw))
                    print('m_dw : {}'.format(m_dw))
                    print('m_qw : {}'.format(m_qw))
                    print('c : {}'.format(c))
                    print([idx_to_word[i] for i in dw[0, :, 0].tolist()])
                    m.train_on_batch(inputs, to_categorical(a, batch_loader_train.max_num_cand))
                    lr = tf.summary.scalar('learning_rate', LEARNING_RATE)
                    summary = tf.summary.merge_all() 
                    s = sess.run(summary)
                    writer.add_summary(s)
                writer.close()
    
    #print("training ...")
    #num_iter = 0
    #max_acc = 0.
    #deltas = []

    #logger = open(save_path+'/log','a',0)

    #if os.path.isfile('%s/best_model.p'%save_path):
    #    print('loading previously saved model')
    #    m.load_model('%s/best_model.p'%save_path)
    #else:
    #    print('saving init model')
    #    m.save_model('%s/model_init.p'%save_path)
    #    print('loading init model')
    #    m.load_model('%s/model_init.p'%save_path)

    #for epoch in xrange(NUM_EPOCHS):
    #    estart = time.time()
    #    new_max = False

    #    for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, fnames in batch_loader_train:
    #        loss, tr_acc, probs = m.train(dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl)

    #        message = "Epoch %d TRAIN loss=%.4e acc=%.4f elapsed=%.1f" % (
    #                epoch, loss, tr_acc, time.time()-estart)
    #        print message
    #        logger.write(message+'\n')

    #        num_iter += 1
    #        if num_iter % VALIDATION_FREQ == 0:
    #            total_loss, total_acc, n, n_cand = 0., 0., 0, 0.

    #            for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, fnames in batch_loader_val:
    #                outs = m.validate(dw, dt, qw, qt, c, a, 
    #                        m_dw, m_qw, tt, tm, m_c, cl)
    #                loss, acc, probs = outs[:3]

    #                bsize = dw.shape[0]
    #                total_loss += bsize*loss
    #                total_acc += bsize*acc
    #                n += bsize

    #    	val_acc = total_acc/n
    #            if val_acc > max_acc:
    #                max_acc = val_acc
    #                m.save_model('%s/best_model.p'%save_path)
    #                new_max = True
    #            message = "Epoch %d VAL loss=%.4e acc=%.4f max_acc=%.4f" % (
    #                epoch, total_loss/n, val_acc, max_acc)
    #            print message
    #            logger.write(message+'\n')

    #    m.save_model('%s/model_%d.p'%(save_path,epoch))
    #    message = "After Epoch %d: Train acc=%.4f, Val acc=%.4f" % (epoch, tr_acc, val_acc)
    #    print message
    #    logger.write(message+'\n')
    #    
    #    # learning schedule
    #    if epoch >=2:
    #        m.anneal()
    #    # stopping criterion
    #    if not new_max:
    #        break

    #logger.close()
