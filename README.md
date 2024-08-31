# NLP Project

This project contains various modules and code related to Natural Language Processing (NLP) tasks. Below is an overview of the different folders and their purposes.

### 1. **Named-entity recognition (NER)**
   This folder contains code for Named Entity Recognition (NER), a task that involves identifying and categorizing entities (like names of people, organizations, locations, etc.) in text. The classifier used in this section is built using the Maximum Entropy method. You will find:
   - **Data**: Sample datasets used for training and testing the NER model.
   - **Models**: Scripts to train the Maximum Entropy classifier.

### 2. **N-gram**
   This folder includes implementations of n-gram models, which are used to predict the next item in a sequence based on the previous n items. It contains:
   - **N-gram generation**: Scripts to generate n-grams from text data.
   - **Vectorization**: Code for vectorizing text using TF-IDF and Bag of Words models.
   - **Bigram and Trigram Extraction**: Scripts for printing bigrams and trigrams that exist in the corpus.

### 3. **Text Classification**
   This folder is focused on text classification tasks, where the goal is to categorize text into predefined classes. Contents include:
   - **Data**: Datasets for training and testing classification models.
   - **Models**: Implementations of classification algorithms including LSI (Latent Semantic Indexing), HDP (Hierarchical Dirichlet Process), and LDA (Latent Dirichlet Allocation).
   - **Evaluation**: Scripts to compare the performance of LSI, HDP, and LDA models.

### 4. **Text Clustering**
   The text clustering folder provides tools for grouping similar texts into clusters. You will find:
   - **Algorithms**: Implementations of clustering algorithms using K-Means and Clarans.
   - **Evaluation**: Scripts to evaluate the performance of the clustering models.
   - **Visualization**: Scripts to visualize the clusters and their characteristics.

### 5. **Text Preprocessing**
   This folder contains code for preprocessing text data before applying NLP models. It includes:
   - **Tokenization**: Scripts for breaking down text into tokens.
   - **Normalization**: Tools for text normalization, such as lowercasing, stemming, and lemmatization.
   - **Stopword Removal**: Code to remove common stopwords from the text.

### 6. **Training a Dependency Parser**
   This folder focuses on training a dependency parser, which analyzes the grammatical structure of a sentence. Contents include:
   - **Data**: Annotated corpora for training the parser.
   - **Models**: Scripts to train the dependency parser.


## Prerequisites

- Python 3.x
- Required libraries (as listed in `requirements.txt`)

## Installation

```bash
pip install -r requirements.txt
```
