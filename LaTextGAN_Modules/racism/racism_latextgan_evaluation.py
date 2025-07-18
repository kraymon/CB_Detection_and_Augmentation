# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import gensim 
from nltk.translate.bleu_score import corpus_bleu

# Imports for latent space analysis
from bokeh.models import Title
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Colorblind3   
from bokeh.models import ColumnDataSource  
from bokeh.transform import factor_cmap   
from bokeh.io import output_notebook        



def tweet_generator(generator, autoencoder, word2vec_model: gensim.models.word2vec.Word2Vec, num_tweets: int=1):
  """Function that generates a given amount of tweets.
  
  Arguments:
    generator (Generator): Generator class instance
    autoencoder (Autoencoder): Autoencoder class instance
    word2vec_model : Pretrained glove model converted to word2vec format
    num_tweets (int): Number of tweets that should be generated
  """

  print()
  average_length = 0
  for _ in range(num_tweets):
    noise = tf.random.normal([1, 100])
    tokens = [word2vec_model.index_to_key[i.numpy()[0] -1] for i in autoencoder.Decoder.inference_mode(states=generator(noise), training=False) if i.numpy()[0] != 0]
    average_length += len(tokens)
    print(f"{' '.join(tokens)}")
    print()
  print()
  print(f"Average length of generated tweets: {average_length/num_tweets} tokens")
  print()



def bleu4_score(generator, autoencoder, word2vec_model: gensim.models.word2vec.Word2Vec, reference_data: [[int]], num_tweets: int=1):
  """Function that calculates the bleu4 score for a given amount of generated tweets.
  
  Arguments:
    generator (Generator): Generator class instance
    autoencoder (Autoencoder): Autoencoder class instance
    word2vec_model : Pretrained glove model converted to word2vec format
    reference_data (list): List containing the reference data for the bleu score computation
    num_tweets (int): Number of tweets that should be generated
  """
  
  generated_tweet = []
  for _ in range(num_tweets):
    noise = tf.random.normal([1, 100])
    generated_tweet.append([word2vec_model.index_to_key[i.numpy()[0] -1] for i in autoencoder.Decoder.inference_mode(states=generator(noise), training=False) if i.numpy()[0] != 0])
  
  bleu_reference=[tweet for _, tweet, _  in reference_data]
  
  hyp = generated_tweet
  
  score_bleu = corpus_bleu([bleu_reference for i in range(500)], hyp, weights=(.25, .25, .25, .25), smoothing_function=None)
  
  return score_bleu



output_notebook()

def latent_space_analysis(generator, autoencoder, train_dataset: tf.data.Dataset, name: str):
  """Plot 2D TSNE Embedding of Generator against Encoder.

  Arguments:
    generator (Generator): Generator class instance
    autoencoder (AutoEncoder): AutoEncoder class instance
    train_dataset (tf.data.Dataset): Dataset to be fed into the Encoder for latent space analysis 
    name (str): Used for in the title of the plot
  """

  # Create a list of real tweet encodings from Encoder
  train_tweets_embeddings = [autoencoder.Encoder(tweet, training=False) for tweet, _, _ in train_dataset.take(500)]
  train_tweets_embeddings = [tweet for tweet_batch in train_tweets_embeddings for tweet in tweet_batch]

  # Create a list of fake tweet encodings from Generator  
  generator_tweets_embeddings=[]
  for _ in range(500):
    noise = tf.random.normal([50, 100])
    generator_tweets_embeddings.append(generator(noise))
  generator_tweets_embeddings = [tweet for tweet_batch in generator_tweets_embeddings for tweet in tweet_batch]


  pca = PCA(n_components=50, svd_solver="randomized", random_state=0)
  pca_embedding_enc = pca.fit_transform(train_tweets_embeddings)
  print('Cumulative explained variation for enocder embedding: {}'.format(np.sum(pca.explained_variance_ratio_)))
  pca_embedding_gen = pca.fit_transform(generator_tweets_embeddings)
  print('Cumulative explained variation for generator embedding: {}'.format(np.sum(pca.explained_variance_ratio_)))
  
  # We apply the TSNE algorithm from scikit to get a 2D embedding of our latent space
  # Once for the Encoder
  tsne = TSNE(n_components=2, perplexity=30., random_state=0)
  tsne_embedding_enc = tsne.fit_transform(pca_embedding_enc)
  
  # Once for the Generator
  tsne_embedding_gen = tsne.fit_transform(pca_embedding_gen)

  # Plotting the TSNE embeddings
  labels =  ["Encoder" for _ in range(len(train_tweets_embeddings))]
  labels.extend(["Generator" for _ in range(len(generator_tweets_embeddings))])

  p = figure(tools="pan,wheel_zoom,reset,save",
            toolbar_location="above",
            title=f"2D Encoder and Generator Embeddings.")
  p.title.text_font_size = "25px"
  p.add_layout(Title(text=name, text_font_size="15px"), 'above')

  x1=np.concatenate((tsne_embedding_enc[:,0], tsne_embedding_gen[:,0]))
  x2=np.concatenate((tsne_embedding_enc[:,1], tsne_embedding_gen[:,1]))

  # Create column dataset from the tsne embedding and labels
  source = ColumnDataSource(data=dict(x1=x1,
                                      x2=x2,
                                      names=labels))

  # Create a scatter plot from the column dataset above
  p.scatter(x="x1", y="x2", size=1, source=source, fill_color=factor_cmap('names', palette=Colorblind3, factors=["Encoder", "Generator"]), fill_alpha=0.3, line_color=factor_cmap('names', palette=Colorblind3, factors=["Encoder", "Generator"]), legend_field='names')  

  show(p)