# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
tfkl = tf.keras.layers



class Encoder(Model):
  # We decided to use type hints whenever the user has the possibility to provide input
  def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=200):
    """Initialize the Encoder that creates an embedding of tweets

    Arguments:
      vocab_size (int): Defines the input dimensionality of the embedding layer
      embedding_matrix (ndarray): Sets the weights of the embedding layer
      bidirectional (bool): Whether to use bidirectional LSTMs or not
      embedding_size (int): Defines the output dimensionality of the embedding layer
    """ 

    super(Encoder, self).__init__()

    # We decided to pretrain our embeddings and therefore set the "trainable" parameter to False due to the limited amount of data at hand.
    # The mask_zero argument is set to True such that the layer can work with padded batches
    self.embedding = tfkl.Embedding(input_dim=vocab_size+1, output_dim=embedding_size, weights=[embedding_matrix], trainable=False, mask_zero=True)
    self.dropout_embed = tfkl.Dropout(0.05)
    self.lstm = tfkl.LSTM(units=100)
    self.norm_post_lstm = tfkl.LayerNormalization()
    self.dropout_after_lstm = tfkl.Dropout(0.1)
    self.dense = tfkl.Dense(600)



  @tf.function(experimental_relax_shapes=True)
  def call(self, x, training: bool=True):

    x = self.embedding(x)
    x = self.dropout_embed(x, training=training)

    hs = self.lstm(x, training=training)
    hs = self.norm_post_lstm(hs)
    hs = self.dropout_after_lstm(hs, training=training)

    dense_out = self.dense(hs, training=training)

    return dense_out



class Decoder(Model):
  def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=200):

    super(Decoder, self).__init__()

    self.embedding = tfkl.Embedding(input_dim=vocab_size+1, output_dim=embedding_size, weights=[embedding_matrix], trainable=False, mask_zero=True)
    self.lstm = tfkl.LSTM(units=600, return_sequences=True, return_state=True)
    self.dropout_post_norm = tfkl.Dropout(0.1)
    self.dense = tfkl.Dense(units=vocab_size+1, activation="softmax")



  @tf.function(experimental_relax_shapes=True)  
  def call(self, x, states, training: bool=True):
    """Activate our Decoder propagating the input through it by using teacher forcing

    Arguments:
      x (tensor): Tensor containing the teacher input
      states(tensor): Tensor containing the last hidden state of the LSTM from the Encoder 
      training (bool): Indicates whether regularization methods should be used or not when calling the Decoder 

    Returns:
      dense_out (tensor): Tensor containing the reconstructed input
    """

    x = self.embedding(x)
    hidden_states, _, _ = self.lstm(x, initial_state=[states, tf.zeros_like(states)], training=training)
    dense_out = self.dense(hidden_states, training=training)
    return dense_out



  def inference_mode(self, states, training: bool=True):
    """Call Decoder in inference mode: Reconstructing the input using only start token and embeddings. 
    Each Decoder step gets the previous prediction of the Decoder as additional input.

    Arguments:
      states(tensor): Tensor containing the last hidden state of the LSTM from the Encoder
      training (bool): Indicates whether regularization methods should be used or not when calling the Encoder 

    Returns:
      predictions (List): List containing the reconstructed input
    """
    # dont forget to change start token and end token to the correct values ! 
    predictions = []
    start_token = self.embedding(tf.constant([[3400]]))
    _, hs, cs = self.lstm(start_token, initial_state=[states, tf.zeros_like(states)], training=training)
    dense_out = self.dense(hs, training=training)
    pred = tf.argmax(dense_out, output_type=tf.int32,  axis=1)
    predictions.append(pred)
    max_seq_length = 35
    end_token = 3401
    stopping_criterion = False


    # The stopping criterion is used to tell the Decoder when to stop predicting new tokens.
    # (either when eos token has been generated or the max sequence length has been reached)
    while not stopping_criterion:

      last_pred = self.embedding(tf.expand_dims(pred, axis=0))
      _, hs, cs = self.lstm(last_pred, initial_state=[hs, cs], training=training) 
      dense_out = self.dense(hs, training=training)
      pred = tf.argmax(dense_out, output_type=tf.int32,  axis=1) 
      predictions.append(pred)
      

      if pred == end_token or len(predictions) >= max_seq_length:
        stopping_criterion=True


    return predictions



class AutoEncoder(Model):

  def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=200):
    """Initialize an Autoencoder consisting of an Encoder and Decoder

    Arguments:
      vocab_size (int): Defines the input dimensionality of the embedding layer
      embedding_matrix (ndarray): Sets the weights of the embedding layer
      embedding_size (int): Defines the output dimensionality of the embedding layer
    """  

    super(AutoEncoder, self).__init__()

    self.Encoder = Encoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, embedding_size=embedding_size)
    self.Decoder = Decoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, embedding_size=embedding_size)


  @tf.function(experimental_relax_shapes=True)    
  def call(self, input, teacher, training: bool=True):
    """Activate our Autoencoder propagating the input through the Encoder and Decoder respectively

    Arguments:
      input (tensor): Tensor containing the input to the Encoder
      teacher (tensor): Tensor containing the input to the Decoder
      training (bool): Indicates whether regularization methods should be used or not when calling the Autoencoder 

    Returns:
      predictions (tensor): Tensor containing the reconstructed input
    """

    hs = self.Encoder(input, training=training)
    predictions = self.Decoder(teacher, states=hs, training=training)
    return predictions