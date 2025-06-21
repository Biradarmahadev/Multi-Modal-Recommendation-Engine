# Multi-Modal Recommendation System for Streaming Platforms

This project implements a multi-modal recommendation system for streaming platforms that combines different types of data to provide personalized content recommendations.

## Theoretical Background

### 1. Recommendation Systems Overview
Recommendation systems are algorithms designed to suggest relevant items to users based on their preferences, behaviors, and item characteristics. They are widely used in streaming platforms, e-commerce, and content delivery systems.

### 2. Multi-Modal Approach
This project implements a hybrid recommendation system that combines multiple approaches:

#### a) Collaborative Filtering
- **Concept**: Predicts user preferences by collecting preferences from many users (the collaborative part).
- **Types**:
  - User-based: Finds similar users and recommends items they liked
  - Item-based: Finds similar items based on user interactions
- **Implementation**: Uses Alternating Least Squares (ALS) for implicit feedback

#### b) Content-Based Filtering
- **Concept**: Recommends items similar to those a user liked in the past, based on item features.
- **Features Used**:
  - Text data (titles, descriptions)
  - Categorical data (genres, categories)
  - Numerical data (ratings, release years)
- **Implementation**: Uses TF-IDF for text and MinMax scaling for numerical features

### 3. Hybrid Recommendation System
- **Advantages**:
  - Overcomes limitations of individual approaches
  - Provides more accurate and diverse recommendations
  - Better handles the cold-start problem
- **Implementation**: Combines scores from both collaborative and content-based approaches

### 4. Evaluation Metrics
- **Precision@K**: Proportion of relevant items in the top-K recommendations
- **Recall@K**: Proportion of relevant items found in the top-K recommendations
- **NDCG@K**: Considers the position of relevant items in the recommendation list

## Features

- **Content-based Filtering**: Utilizes item features like text, categories, and metadata
- **Collaborative Filtering**: Implements user-item interaction analysis using ALS
- **Hybrid Approach**: Combines multiple recommendation techniques for better accuracy
- **Feature Engineering**: Advanced processing of text and numerical features
- **Evaluation Metrics**: Includes precision@K, recall@K, and NDCG for model assessment
- **Sample Dataset**: Comes with a pre-processed dataset for quick start

## Project Structure

```
├── data/                    # Data directory
│   ├── raw/                 # Raw datasets
│   └── processed/           # Processed datasets
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model implementations
│   ├── evaluation/          # Evaluation metrics
│   └── utils/               # Utility functions
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the data preprocessing pipeline:
   ```
   python src/data/preprocess.py
   ```
2. Train the recommendation model:
   ```
   python src/models/train.py
   ```
3. Generate recommendations:
   ```
   python src/models/predict.py
   ```

## Dataset

This project uses the [MovieLens](https://grouplens.org/datasets/movielens/) dataset for demonstration purposes. The dataset includes:

- User ratings
- Movie metadata
- User demographics
- Movie genres and tags

## License

MIT
