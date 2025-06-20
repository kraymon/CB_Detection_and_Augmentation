### Context
This repository provides a framework for cyberbullying detection and data augmentation. It is structured in four main phases:
1) Data preprocessing (1_preprocessing)
2) Word embedding comparison (2_Embedding_Comparison)
3) Data augmentation with GANs (3_GAN_notebook)
4) Classification

### Implementation Details
- The dataset used is the twitter_parsed_dataset from Kaggle. : https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset
- We compared five embedding techniques using t-SNE visualization:
  - Bag-of-Words (BoW)
  - TF-IDF
  - SBERT
  - Pretrained GloVe (glove.6B.200d)
  - Word2Vec
    > SBERT and GloVe performed best in capturing class separation.
- For data augmentation, we used LaTextGAN (Donahue & Rumshisky, 2019) with pretrained GloVe embeddings to generate tweets for racism and sexism (the minority classes).
  > We trained two separate LaTextGANs, one per class.
- We adapted and modified the implementation of LaTextGAN published by Gerrit Bartels and Jacob Dudek designed to generate Trump-like tweets. Available here : https://github.com/GerritBartels/LaTextGAN

- Finally, we compared three augmentation strategies (GAN, SMOTE, Random Oversampling) on three classifiers:
  - Decision Tree (DT)
  - Random Forest (RF)
  - LSTM
    > We also tested three sampling approaches:
  - Augmentation before train-test split
  - Augmentation after split
  - 10-Fold Cross-Validation
### Recommendations
- Before running the notebooks, make sure to download the Kaggle dataset and pretrained GloVe embeddings (glove.6B.200d.txt) and place them in the appropriate folders.
- To train LaTextGAN's autoencoder, we recommend using Google Colab due to GPU requirements.
