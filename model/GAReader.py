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
        #if W_init is None: W_init = lasagne.init.GlorotNormal().sample((vocab_size, self.embed_dim))

        #doc_var, query_var, cand_var = T.itensor3('doc'), T.itensor3('quer'), \
        #        T.wtensor3('cand')
        #docmask_var, qmask_var, candmask_var = T.bmatrix('doc_mask'), T.bmatrix('q_mask'), \
        #        T.bmatrix('c_mask')
        #target_var = T.ivector('ans')
        #feat_var = T.imatrix('feat')
        #doc_toks, qry_toks= T.imatrix('dchars'), T.imatrix('qchars')
        #tok_var, tok_mask = T.imatrix('tok'), T.bmatrix('tok_mask')
        #cloze_var = T.ivector('cloze')
        #self.inps = [doc_var, doc_toks, query_var, qry_toks, cand_var, target_var, docmask_var,
        #        qmask_var, tok_var, tok_mask, candmask_var, feat_var, cloze_var]

        #self.predicted_probs, predicted_probs_val, self.network, W_emb, attentions = (
        #        self.build_network(K, vocab_size, W_init))

        #self.loss_fn = tf.keras.backend.categorical_crossentropy(target_var, self.predicted_probs).mean()
        #self.eval_fn = tf.keras.metrics.categorical_accuracy(target_var, self.predicted_probs, 
        #        target_var).mean()

        #loss_fn_val = tf.keras.backend.categorical_crossentropy(target_var, predicted_probs_val).mean()
        #eval_fn_val = tf.keras.metrics.categorical_accuracy(target_var, predicted_probs_val).mean()

        #self.params = L.get_all_params(self.network, trainable=True)
        #
        #updates = lasagne.updates.adam(self.loss_fn, self.params, learning_rate=self.learning_rate)

        #self.train_fn = theano.function(self.inps,
        #        [self.loss_fn, self.eval_fn, self.predicted_probs], 
        #        updates=updates,
        #        on_unused_input='warn')
        #self.validate_fn = theano.function(self.inps, 
        #        [loss_fn_val, eval_fn_val, predicted_probs_val]+attentions,
        #        on_unused_input='warn')

    def anneal(self):
        self.learning_rate /= 2
        updates = lasagne.updates.adam(self.loss_fn, self.params, learning_rate=self.learning_rate)
        self.train_fn = theano.function(self.inps, \
                [self.loss_fn, self.eval_fn, self.predicted_probs], 
                updates=updates,
                on_unused_input='warn')

    def train(self, dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl):
        f = prepare_input(dw,qw)
        return self.train_fn(dw, dt, qw, qt, c, a, 
                m_dw.astype('int8'), m_qw.astype('int8'), 
                tt, tm.astype('int8'), 
                m_c.astype('int8'), f, cl)

    def validate(self, dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl):
        f = prepare_input(dw,qw)
        return self.validate_fn(dw, dt, qw, qt, c, a, 
                m_dw.astype('int8'), m_qw.astype('int8'), 
                tt, tm.astype('int8'), 
                m_c.astype('int8'), f, cl)

    def build_network(self):
        l_docin = tf.keras.layers.Input(shape=(None,), dtype=tf.int64)
        #l_doctokin = L.InputLayer(shape=(None,None), input_var=self.inps[1])
        l_qin = tf.keras.layers.Input(shape=(None,), dtype=tf.int64)
        #l_qtokin = L.InputLayer(shape=(None,None), input_var=self.inps[3])
        l_docmask = tf.keras.layers.Input(shape=(None,), dtype=tf.int64)
        l_qmask = tf.keras.layers.Input(shape=(None,), dtype=tf.int64)
        #l_tokin = L.InputLayer(shape=(None,MAX_WORD_LEN), input_var=self.inps[8])
        #l_tokmask = L.InputLayer(shape=(None,MAX_WORD_LEN), input_var=self.inps[9])
        l_featin = tf.keras.layers.Input(shape=(None,), dtype=tf.int64)

        cand_var = tf.keras.layers.Input(shape=(None, None), dtype=tf.int64)
        cloze_var = tf.keras.layers.Input(shape=(1,), dtype=tf.int64)
        candmask_var = tf.keras.layers.Input(shape=(None,), dtype=tf.int64)

        doc_shp = tf.shape(l_docin)
        qry_shp = tf.shape(l_qin)

        l_docembed = tf.keras.layers.Embedding(input_dim=self.vocab_size, 
                output_dim=self.embed_dim, embeddings_initializer=tf.constant_initializer(self.W_init), mask_zero=True)(l_docin) # B x N x 1 x DE
        l_docebed = tf.keras.layers.Lambda(lambda x: x, output_shape=lambda s:s)(l_docembed)
        l_doce = tf.keras.layers.Reshape(
                (doc_shp[1],self.embed_dim))(l_docembed) # B x N x DE
        l_qemb = tf.keras.layers.Embedding(input_dim=self.vocab_size, 
                output_dim=self.embed_dim, embeddings_initializer=tf.constant_initializer(self.W_init), mask_zero=True)(l_qin)
        l_qemb = tf.keras.layers.Lambda(lambda x: x, output_shape=lambda s:s)(l_qemb)
        l_qembed = tf.keras.layers.Reshape(
                (qry_shp[1],self.embed_dim))(l_qemb) # B x N x DE
        l_fembed = tf.keras.layers.Embedding(input_dim=2, output_dim=2)(l_featin) # B x N x 2

        #if self.train_emb==0: 
        #    l_docembed.params[l_docembed.W].remove('trainable')
        #    l_qemb.params[l_qemb.W].remove('trainable')

        # char embeddings
        #if self.use_chars:
        #    l_lookup = L.EmbeddingLayer(l_tokin, self.num_chars, self.char_dim) # T x L x D
        #    l_fgru = L.GRULayer(l_lookup, self.char_dim, grad_clipping=GRAD_CLIP, 
        #            mask_input=l_tokmask, gradient_steps=GRAD_STEPS, precompute_input=True,
        #            only_return_final=True)
        #    l_bgru = L.GRULayer(l_lookup, self.char_dim, grad_clipping=GRAD_CLIP, 
        #            mask_input=l_tokmask, gradient_steps=GRAD_STEPS, precompute_input=True, 
        #            backwards=True, only_return_final=True) # T x 2D
        #    l_fwdembed = L.DenseLayer(l_fgru, self.embed_dim/2, nonlinearity=None) # T x DE/2
        #    l_bckembed = L.DenseLayer(l_bgru, self.embed_dim/2, nonlinearity=None) # T x DE/2
        #    l_embed = L.ElemwiseSumLayer([l_fwdembed, l_bckembed], coeffs=1)
        #    l_docchar_embed = IndexLayer([l_doctokin, l_embed]) # B x N x DE/2
        #    l_qchar_embed = IndexLayer([l_qtokin, l_embed]) # B x Q x DE/2

        #    l_doce = L.ConcatLayer([l_doce, l_docchar_embed], axis=2)
        #    l_qembed = L.ConcatLayer([l_qembed, l_qchar_embed], axis=2)

        #attentions = []
        #if self.save_attn:
        #    l_m = PairwiseInteractionLayer([l_doce,l_qembed])
        #    attentions.append(L.get_output(l_m, deterministic=True))

        for i in range(self.K-1):

            l_doce = tf.keras.layers.Masking(mask_value=0.)(l_doce)
            l_fwd_doc_1 = tf.keras.layers.GRU(self.nhidden, return_sequences=True)(l_doce)
            l_bkd_doc_1 = tf.keras.layers.GRU(self.nhidden, go_backwards=True, 
                    return_sequences=True)(l_doce)

            print('l_bkd_doc_1 ' + str(l_bkd_doc_1))
            l_doc_1 = tf.keras.layers.Concatenate(axis=2)([l_fwd_doc_1, l_bkd_doc_1]) # B x N x DE
            print('l_doc_1 ' + str(l_doc_1))

            l_qembed = tf.keras.layers.Masking(mask_value=0.)(l_qembed)
            l_fwd_q_1 = tf.keras.layers.GRU(self.nhidden, return_sequences=True)(l_qembed)
            l_bkd_q_1 = tf.keras.layers.GRU(self.nhidden, go_backwards=True, 
                    return_sequences=True)(l_qembed)

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
            #if self.save_attn: 
            #    attentions.append(L.get_output(l_m, deterministic=True))

        #if self.use_feat: l_doce = tf.keras.layers.Concatenate(axis=2)([l_doce, l_fembed])# B x N x DE+2

        # final layer
        l_doce = tf.keras.layers.Masking(mask_value=0.)(l_doce)
        l_fwd_doc = tf.keras.layers.GRU(self.nhidden, return_sequences=True)(l_doce)
        l_bkd_doc = tf.keras.layers.GRU(self.nhidden, go_backwards=True, 
                return_sequences=True)(l_doce)
        l_doc = tf.keras.layers.Concatenate(axis=2)([l_fwd_doc, l_bkd_doc])

        l_qembed = tf.keras.layers.Masking(mask_value=0.)(l_qembed)
        l_fwd_q = tf.keras.layers.GRU(self.nhidden, return_sequences=True)(l_qembed)
        l_bkd_q = tf.keras.layers.GRU(self.nhidden, go_backwards=True, 
                return_sequences=True)(l_qembed)
        l_q = tf.keras.layers.Concatenate(axis=2)([l_fwd_q, l_bkd_q]) # B x Q x 2D

        #if self.save_attn:
        #    l_m = PairwiseInteractionLayer()([l_doc, l_q])
        #    attentions.append(L.get_output(l_m, deterministic=True))

        l_doc = tf.keras.layers.Lambda(lambda x: x, output_shape=lambda s:s)(l_doc)
        l_q = tf.keras.layers.Lambda(lambda x: x, output_shape=lambda s:s)(l_q)
        l_prob = AttentionSumLayer(cand_var, cloze_var, mask_input=candmask_var)([l_doc, l_q])
        #final = L.get_output(l_prob)
        #final_v = L.get_output(l_prob, deterministic=True)

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
