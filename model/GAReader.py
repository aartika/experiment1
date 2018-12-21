import numpy as np
import cPickle as pickle
from config import *
from layers import *

def prepare_input(d,q):
    f = np.zeros(d.shape[:2]).astype('int32')
    for i in range(d.shape[0]):
        f[i,:] = np.in1d(d[i,:,0],q[i,:,0])
    return f

class Model():

    def __init__(self, K, vocab_size, num_chars, W_init, 
            nhidden, embed_dim, dropout, train_emb, char_dim, use_feat, gating_fn, 
            words,
            save_attn=False):
        self.nhidden = nhidden
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.train_emb = train_emb
        self.char_dim = char_dim
        self.learning_rate = LEARNING_RATE
        self.num_chars = num_chars
        self.use_feat = use_feat
        self.save_attn = save_attn
        self.gating_fn = gating_fn
        self.vocab_size = vocab_size
        self.W_init = W_init
        self.K = K

        self.use_chars = self.char_dim!=0
        self.mapping_string = tf.constant(words)

    def build_network(self, max_doc_len, max_qry_len, max_num_cand):
        l_docin = tf.keras.layers.Input(shape=(max_doc_len, 1))
        l_qin = tf.keras.layers.Input(shape=(max_qry_len, 1))
        l_docmask = tf.keras.layers.Input(shape=(max_doc_len,))
        l_qmask = tf.keras.layers.Input(shape=(max_qry_len,))
        l_featin = tf.keras.layers.Input(shape=(None,))

        cand_var = tf.keras.layers.Input(shape=(max_doc_len, max_num_cand))
        cloze_var = tf.keras.layers.Input(shape=(1,))
        candmask_var = tf.keras.layers.Input(shape=(max_doc_len,))

        doc_shp = tf.shape(l_docin)
        qry_shp = tf.shape(l_qin)

        l_docembed = tf.keras.layers.Embedding(input_dim=self.vocab_size, 
                output_dim=self.embed_dim, embeddings_initializer=tf.constant_initializer(self.W_init), mask_zero=True)(l_docin) # B x N x 1 x DE
        l_docembed = tf.keras.layers.Lambda(lambda x: x, output_shape=lambda s:s)(l_docembed)
        l_doce = tf.keras.layers.Reshape(
                (doc_shp[1],self.embed_dim))(l_docembed) # B x N x DE
        l_qemb = tf.keras.layers.Embedding(input_dim=self.vocab_size, 
                output_dim=self.embed_dim, embeddings_initializer=tf.constant_initializer(self.W_init), mask_zero=True)(l_qin)
        l_qemb = tf.keras.layers.Lambda(lambda x: x, output_shape=lambda s:s)(l_qemb)
        l_qembed = tf.keras.layers.Reshape(
                (qry_shp[1],self.embed_dim))(l_qemb) # B x N x DE
        l_fembed = tf.keras.layers.Embedding(input_dim=2, output_dim=2)(l_featin) # B x N x 2
        
        normal = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)

        for i in range(self.K-1):

            l_doce = tf.keras.layers.Masking(mask_value=0.)(l_doce)
            l_fwd_doc_1 = tf.keras.layers.GRU(self.nhidden, return_sequences=True, implementation=2)(l_doce)
            l_bkd_doc_1 = tf.keras.layers.GRU(self.nhidden, go_backwards=True, 
                    return_sequences=True, implementation=2)(l_doce)

            print('l_bkd_doc_1 ' + str(l_bkd_doc_1))
            l_doc_1 = tf.keras.layers.Concatenate(axis=2)([l_fwd_doc_1, l_bkd_doc_1]) # B x N x DE
            print('l_doc_1 ' + str(l_doc_1))

            l_qembed = tf.keras.layers.Masking(mask_value=0.)(l_qembed)
            l_fwd_q_1 = tf.keras.layers.GRU(self.nhidden, return_sequences=True, implementation=2)(l_qembed)
            l_bkd_q_1 = tf.keras.layers.GRU(self.nhidden, go_backwards=True, 
                    return_sequences=True, implementation=2)(l_qembed)

            print('l_bkd_q_1 ' + str(l_bkd_q_1))

            l_q_c_1 = tf.keras.layers.Concatenate(axis=2)([l_fwd_q_1, l_bkd_q_1]) # B x Q x DE
            print('l_q_c_1 ' + str(l_q_c_1))

            l_doc_1 = tf.keras.layers.Lambda(lambda x: x, output_shape=lambda s:s)(l_doc_1)
            l_q_c_1 = tf.keras.layers.Lambda(lambda x: x, output_shape=lambda s:s)(l_q_c_1)
            l_m = PairwiseInteractionLayer()([l_doc_1, l_q_c_1])
            l_doc_2_in = GatedAttentionLayer(
                    gating_fn=self.gating_fn, 
                    mask_input=l_qmask)([l_doc_1, l_q_c_1, l_m])
            l_doce = tf.keras.layers.Dropout(rate=self.dropout)(l_doc_2_in) # B x N x DE

        # final layer
        l_doce = tf.keras.layers.Masking(mask_value=0.)(l_doce)
        l_fwd_doc = tf.keras.layers.GRU(self.nhidden, return_sequences=True, implementation=2)(l_doce)
        l_bkd_doc = tf.keras.layers.GRU(self.nhidden, go_backwards=True, 
                return_sequences=True, implementation=2)(l_doce)
        l_doc = tf.keras.layers.Concatenate(axis=2)([l_fwd_doc, l_bkd_doc])

        l_qembed = tf.keras.layers.Masking(mask_value=0.)(l_qembed)
        l_fwd_q = tf.keras.layers.GRU(self.nhidden, return_sequences=True, implementation=2)(l_qembed)
        l_bkd_q = tf.keras.layers.GRU(self.nhidden, go_backwards=True, 
                return_sequences=True, implementation=2)(l_qembed)
        l_q = tf.keras.layers.Concatenate(axis=2)([l_fwd_q, l_bkd_q]) # B x Q x 2D

        l_doc = tf.keras.layers.Lambda(lambda x: x, output_shape=lambda s:s)(l_doc)
        l_q = tf.keras.layers.Lambda(lambda x: x, output_shape=lambda s:s)(l_q)
        l_prob = AttentionSumLayer(cand_var, cloze_var, mask_input=candmask_var)([l_doc, l_q])

        return tf.keras.Model(inputs=[l_docin, l_qin, l_docmask, l_qmask,
            cand_var, candmask_var, cloze_var], outputs=l_prob)

    def load_model(self, load_path):
        with open(load_path, 'r') as f:
            data = pickle.load(f)
        L.set_all_param_values(self.network, data)

    def save_model(self, save_path):
        data = L.get_all_param_values(self.network)
        with open(save_path, 'w') as f:
            pickle.dump(data, f)
