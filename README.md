# Word Embeddings: From Scratch to Pretrained Models

## 📌 Project Overview

This project explores word embedding techniques in Natural Language Processing (NLP).

It includes:
- Word2Vec CBOW implementation (TensorFlow)
- Word2Vec Skip-gram with negative sampling
- FastText pretrained embeddings
- Text preprocessing pipeline
- Similarity and analogy evaluation

The goal is to understand how distributed word representations are learned and how semantic relationships emerge from training.

---

## 🧠 Implemented Models

### 1️⃣ CBOW (Continuous Bag of Words)
- Predicts a target word from its context
- Full softmax output layer
- Embedding dimension: 100
- Trained on the cleaned text8 corpus

### 2️⃣ Skip-gram with Negative Sampling
- Predicts context words from a target word
- Uses negative sampling for efficiency
- Includes subsampling of frequent words
- More scalable than full softmax

### 3️⃣ FastText (Pretrained)
- Loaded using Gensim
- Used for similarity and analogy evaluation

---

## 📊 Experiments & Evaluation

The following analyses were performed:

- Word similarity using cosine similarity
- Nearest neighbors search
- Word analogies (e.g., king - man + woman)
- Comparison between trained embeddings and pretrained FastText

---

## 📁 Project Structure

- clean_height_weight.ipynb
- clean_text8.ipynb
- word2vec_cbow_tf.ipynb
- word2vec_skipgram_tf.ipynb
- fasttext_pretrained.ipynb
- Rapport_Word_Embeddings.ipynb


---

## 🛠 Requirements

- Python 3.9+
- TensorFlow
- NumPy
- scikit-learn
- NLTK
- Gensim
- Matplotlib

Install dependencies:

```bash
pip install tensorflow numpy scikit-learn nltk gensim matplotlib
