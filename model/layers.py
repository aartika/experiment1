import numpy as np
import tensorflow as tf

def Tconcat(t1,t2):
    return T.concatenate([t1,t2], axis=2)

def Tsum(t1,t2):
    return t1+t2

class GatedAttentionLayer(tf.keras.layers.Layer):
    """
    Layer which gets two 3D tensors as input, and a pairwise matching matrix M between 
    the second dimension of each (with the third dimension as features), and gates each 
    element in the first tensor by a weighted average vector from the other tensor. The weights 
    are a softmax over the pairwise matching matrix. The gating function is specified at input.
    The mask is for the second tensor.
    """

    def __init__(self, gating_fn='T.mul', mask_input=None, transpose=False, **kwargs):
        super(GatedAttentionLayer, self).__init__(**kwargs)
	self.gating_fn = gating_fn
        #if mask_input is not None and type(mask_input).__name__!='TensorVariable': 
        #    raise TypeError('Mask input must be theano tensor variable')
        self.mask = mask_input
        self.transpose = transpose

    def compute_output_shape(self, input_shapes):
        if self.gating_fn=='Tconcat': 
            return (input_shapes[0][0],input_shapes[0][1],input_shapes[0][2]+input_shapes[1][2])
        else:
            return input_shapes[0]

    def call(self, inputs):

        # inputs[0]: B x N x D
        # inputs[1]: B x Q x D
        # inputs[2]: B x N x Q / B x Q x N
        # self.mask: B x Q

        if self.transpose: M = tf.transpose(inputs[2], (0, 2, 1))
        else: M = inputs[2]
        print('inputs[2] ' + str(inputs[2]))
        M_shape = tf.shape(M)
        alphas = tf.nn.softmax(tf.reshape(M, [M_shape[0]*M_shape[1],M_shape[2]]))
        print('alphas ' + str(alphas))
        print('M_shape ' + str(M_shape))
        alphas = tf.reshape(alphas, M_shape)
        print('alphas ' + str(alphas))
        print('self.mask ' + str(self.mask))
        alphas_r = tf.multiply(alphas, self.mask[:,np.newaxis,:]) # B x N x Q
        print('alphas_r ' + str(alphas_r))
        alphas_r = tf.divide(alphas_r, tf.keras.backend.sum(alphas_r, axis=2)[:,:,np.newaxis]) # B x N x Q
        print('alphas_r ' + str(alphas_r))
        q_rep = tf.keras.backend.batch_dot(alphas_r, inputs[1]) # B x N x D
    
        print('q_rep ' + str(q_rep))
        return eval(self.gating_fn)([inputs[0],q_rep], 2)

class PairwiseInteractionLayer(tf.keras.layers.Layer):
    """
    Layer which gets two 3D tensors as input, computes pairwise matching matrix M between 
    the second dimension of each (with the third dimension as features). 
    """

    def __init__(self, **kwargs):
        super(PairwiseInteractionLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], input_shapes[1][1])

    def call(self, inputs):

        # inputs[0]: B x N x D
        # inputs[1]: B x Q x D
        # self.mask: B x Q

        print(inputs[0])
        print(inputs[1])
        q_shuf = tf.transpose(inputs[1], (0, 2, 1)) # B x D x Q
        print(q_shuf)
        return tf.keras.backend.batch_dot(inputs[0], q_shuf, axes=[2, 1]) # B x N x Q

class AttentionSumLayer(tf.keras.layers.Layer):
    """
    Layer which takes two 3D tensors D,Q, an aggregator A, and a pointer X as input. First elements
    of Q indexed by X are extracted, then a matching score between D and the extracted element is 
    computed. Finally the scores are aggregated by multiplying with A and returned. The mask input
    is over D.
    """

    def __init__(self, aggregator, pointer, mask_input=None, **kwargs):
        super(AttentionSumLayer, self).__init__(**kwargs)
        #if mask_input is not None and type(mask_input).__name__!='TensorVariable': 
        #    raise TypeError('Mask input must be theano tensor variable')
        self.mask = mask_input
        self.aggregator = aggregator
        self.pointer = tf.cast(pointer, tf.int32)
        
    def compute_output_shape(self, input_shapes):
        return (input_shapes[2][0], input_shapes[2][2])

    def call(self, inputs):

        # inputs[0]: B x N x D
        # inputs[1]: B x Q x D
        # self.aggregator: B x N x C
        # self.pointer: B x 1
        # self.mask: B x N

        batch_size = tf.shape(inputs[1])[0]
        indices = tf.concat([tf.reshape(tf.range(batch_size), [batch_size, 1]), self.pointer], axis=1)
        print('indices ' + str(indices))
        q = tf.gather_nd(inputs[1], indices) # B x D
        p = tf.keras.backend.batch_dot(inputs[0], q) # B x N
        p = tf.Print(p, [tf.shape(p), tf.shape(self.mask), tf.shape(inputs[0]), tf.shape(self.aggregator)])
        pm = tf.nn.softmax(p)*self.mask # B x N
        pm = pm/tf.keras.backend.sum(pm, axis=1)[:,np.newaxis] # B x N

        return tf.keras.backend.batch_dot(pm, self.aggregator)

#class BilinearAttentionLayer(L.MergeLayer):
#    """
#    Layer which implements the bilinear attention described in Stanfor AR (Chen, 2016).
#    Takes a 3D tensor P and a 2D tensor Q as input, outputs  a 2D tensor which is Ps 
#    weighted average along the second dimension, and weights are q_i^T W p_i attention 
#    vectors for each element in batch of P and Q. 
#    If mask_input is provided it will be applied to the output attention vectors before
#    averaging. Mask input should be theano variable and not lasagne layer.
#    """
#
#    def __init__(self, incomings, num_units, W=lasagne.init.Uniform(), 
#            mask_input=None, **kwargs):
#        super(BilinearAttentionLayer, self).__init__(incomings, **kwargs)
#        self.num_units = num_units
#        if mask_input is not None and type(mask_input).__name__!='TensorVariable': 
#            raise TypeError('Mask input must be theano tensor variable')
#        self.mask = mask_input
#        self.W = self.add_param(W, (num_units, num_units), name='W')
#
#    def get_output_shape_for(self, input_shapes):
#        return (input_shapes[0][0], input_shapes[0][2])
#
#    def get_output_for(self, inputs, **kwargs):
#
#        # inputs[0]: # B x N x H
#        # inputs[1]: # B x H
#        # self.W: H x H
#        # self.mask: # B x N
#
#        qW = T.dot(inputs[1], self.W) # B x H
#        qWp = (inputs[0]*qW[:,np.newaxis,:]).sum(axis=2)
#        alphas = T.nnet.softmax(qWp)
#        if self.mask is not None:
#            alphas = alphas*self.mask
#            alphas = alphas/alphas.sum(axis=1)[:,np.newaxis]
#        return (inputs[0]*alphas[:,:,np.newaxis]).sum(axis=1)
#
#class IndexLayer(L.MergeLayer):
#    """
#    Layer which takes two inputs: a tensor D with arbitrary shape, and integer values,
#    and a 2D lookup tensor whose rows are indices in D. Returns the first tensor with
#    its each value replaced by the lookup from second tensor. This is similar to 
#    EmbeddingLayer, but the lookup matrix is not a parameter, and can be of arbitrary
#    size
#    """
#
#    def get_output_shape_for(self, input_shapes):
#        return input_shapes[0] + (input_shapes[1][-1],)
#
#    def get_output_for(self, inputs, **kwargs):
#        return inputs[1][inputs[0]]
#
