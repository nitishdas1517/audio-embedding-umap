# audio-embedding-umap
Visualization of audio features embedding using UMAP

# Audio Feature Embeddings and Classification with UMAP and Conformer

This repository contains a comprehensive project involving the application of UMAP (Uniform Manifold Approximation and Projection) to understand and visualize audio feature embeddings for classification tasks. Additionally, it includes a review of the Conformer model and the implementation of a keyword spotter using pretrained models.

## Table of Contents
1. [Understanding UMAP](#understanding-umap)
2. [Review of the Conformer Paper](#review-of-the-conformer-paper)
3. [Building the Classifier](#building-the-classifier)
4. [Audio Feature Embeddings](#audio-feature-embeddings)
5. [Analysis and Thoughts](#analysis-and-thoughts)
6. [Getting Started](#getting-started)
7. [Usage](#usage)

## Understanding UMAP
UMAP (Uniform Manifold Approximation and Projection) is a powerful dimensionality reduction technique. It is particularly effective for visualizing high-dimensional data by projecting it into lower dimensions (typically 2D or 3D). In this project, we use UMAP to visualize and understand the classification solutions of various datasets.

### Example Notebook
Section-1 provides an illustrative example of using UMAP with a classification dataset. It demonstrates how UMAP can be used to reduce dimensionality and visualize clusters within the data.

## Review of the Conformer Paper
The Conformer model introduces several key innovations by combining convolutional neural networks with transformers. This hybrid approach captures both local and global dependencies in sequential data like speech. The main innovations of the Conformer paper include:
1. Integrating convolution layers within the transformer architecture.
2. Utilizing relative positional encoding for better sequence information handling.
3. Achieving state-of-the-art performance on various speech recognition benchmarks.

### Summary Notebook
Section-2 presents a 3-5 point summary of the new ideas and innovations introduced in the Conformer paper.

## Building the Classifier
Using pretrained models such as Wav2Vec2 combined with the Conformer architecture, we build a keyword spotter. The implementation leverages resources from Hugging Face.

### Keyword Spotter Notebook
Section-2 uses the pretrained model from [Hugging Face](https://huggingface.co/juliensimon/wav2vec2-conformer-rel-pos-large-finetuned-speech-commands) to build a classifier for keyword spotting.

## Audio Feature Embeddings
Using the MINDS-14 dataset from [Hugging Face](https://huggingface.co/datasets/PolyAI/minds14), we build a dataset of audio feature embeddings. We then use UMAP to create lower-dimensional space embeddings for visualization and clustering in an unsupervised manner.

### Feature Embeddings Notebook
Section-3 demonstrates how to extract features from the MINDS-14 dataset, apply UMAP for dimensionality reduction, and visualize the embeddings. It also includes clustering using KMeans.

## Analysis and Thoughts
The final part of the project involves a comprehensive analysis of the results obtained from using UMAP for visualization and clustering, as well as the performance of the keyword spotter classifier. This section presents insights and reflections on the entire exercise.


## Getting Started
To get a copy of this project up and running on your local machine, follow these simple steps.

### Prerequisites
- Python 3.6 or later
- Jupyter Notebook or Google Colab
- Required Python packages: `librosa`, `torch`, `matplotlib`, `umap-learn`, `scikit-learn`, `datasets`, `requests`

### Installation
Clone the repository:
```bash
git clone https://github.com/nitishdas1517/audio-embedding-umap.git
```
Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
Open the notebooks in Jupyter or Google Colab and run the cells to reproduce the results.
---

Feel free to reach out with any questions or feedback. Enjoy exploring the fascinating world of UMAP, audio feature embeddings, and the Conformer model!

