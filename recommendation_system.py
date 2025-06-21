import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
from typing import Dict, List, Tuple, Union

class MultiModalRecommender:
    """
    A multi-modal recommender system that combines content-based and collaborative filtering.
    """
    
    def __init__(self, n_recommendations: int = 10):
        """
        Initialize the recommender system.
        
        Args:
            n_recommendations: Number of recommendations to return
        """
        self.n_recommendations = n_recommendations
        self.item_features = None
        self.user_features = None
        self.item_similarity = None
        self.model = None
        self.item_ids = None
        self.user_ids = None
        
    def fit(self, 
             interactions: pd.DataFrame,
             item_features: pd.DataFrame = None,
             user_features: pd.DataFrame = None) -> None:
        """
        Fit the recommendation model.
        
        Args:
            interactions: DataFrame with user-item interactions (columns: user_id, item_id, rating)
            item_features: DataFrame with item features (index: item_id)
            user_features: DataFrame with user features (index: user_id)
        """
        # Store user and item IDs for mapping
        self.user_ids = interactions['user_id'].unique()
        self.item_ids = interactions['item_id'].unique()
        
        # Process item features if provided
        if item_features is not None:
            self._process_item_features(item_features)
        
        # Process user features if provided
        if user_features is not None:
            self._process_user_features(user_features)
            
        # Train collaborative filtering model
        self._train_collaborative_filtering(interactions)
        
    def _process_item_features(self, item_features: pd.DataFrame) -> None:
        """Process and store item features."""
        # Handle text features
        text_columns = item_features.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            tfidf = TfidfVectorizer(stop_words='english')
            text_features = tfidf.fit_transform(item_features[text_columns].fillna('').astype(str).apply(' '.join, axis=1))
        else:
            text_features = None
            
        # Handle numerical features
        num_columns = item_features.select_dtypes(include=['int64', 'float64']).columns
        if len(num_columns) > 0:
            scaler = MinMaxScaler()
            num_features = scaler.fit_transform(item_features[num_columns])
        else:
            num_features = None
            
        # Combine features
        if text_features is not None and num_features is not None:
            self.item_features = hstack([text_features, num_features])
        elif text_features is not None:
            self.item_features = text_features
        elif num_features is not None:
            self.item_features = num_features
            
        # Calculate item similarity matrix if we have item features
        if self.item_features is not None:
            self.item_similarity = cosine_similarity(self.item_features)
    
    def _process_user_features(self, user_features: pd.DataFrame) -> None:
        """Process and store user features."""
        # Similar processing as item features
        # This is a simplified version - you might want to customize this
        # based on your specific user features
        self.user_features = pd.get_dummies(user_features)
    
    def _train_collaborative_filtering(self, interactions: pd.DataFrame) -> None:
        """Train a collaborative filtering model using k-Nearest Neighbors."""
        # Create user-item interaction matrix
        self.user_mapping = {user_id: i for i, user_id in enumerate(self.user_ids)}
        self.item_mapping = {item_id: i for i, item_id in enumerate(self.item_ids)}
        
        # Create user-item matrix
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        # Initialize user-item matrix
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        # Fill the matrix with ratings
        for _, row in interactions.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['item_id']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # Initialize and fit the k-NN model
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
        self.model.fit(self.user_item_matrix)
    
    def recommend(self, 
                 user_id: Union[int, str], 
                 n: int = None,
                 item_features: pd.DataFrame = None) -> List[Tuple[Union[int, str], float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: ID of the user to generate recommendations for
            n: Number of recommendations to return (overrides default if provided)
            item_features: Optional item features for content-based filtering
            
        Returns:
            List of (item_id, score) tuples
        """
        n = n or self.n_recommendations
        
        # Generate collaborative filtering scores
        cf_scores = self._get_cf_scores(user_id)
        
        # If we have item features, generate content-based scores
        if item_features is not None and self.item_features is not None:
            cb_scores = self._get_content_based_scores(user_id, item_features)
            # Combine scores (simple average here, but could be weighted)
            combined_scores = (cf_scores + cb_scores) / 2
        else:
            combined_scores = cf_scores
            
        # Get top N recommendations
        top_indices = np.argsort(combined_scores)[::-1][:n]
        recommendations = [(self.item_ids[i], combined_scores[i]) for i in top_indices]
        
        return recommendations
    
    def _get_cf_scores(self, user_id: Union[int, str]) -> np.ndarray:
        """Get collaborative filtering scores for all items for a user."""
        if user_id not in self.user_mapping:
            return np.zeros(len(self.item_ids))
            
        user_idx = self.user_mapping[user_id]
        user_vector = self.user_item_matrix[user_idx].reshape(1, -1)
        
        # Get distances and indices of nearest neighbors
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=3)
        
        # Average the ratings of nearest neighbors for each item
        neighbor_ratings = self.user_item_matrix[indices[0]]
        avg_ratings = np.mean(neighbor_ratings, axis=0)
        
        # Set already rated items to 0
        user_ratings = self.user_item_matrix[user_idx]
        avg_ratings[user_ratings > 0] = 0
        
        return avg_ratings
    
    def _get_content_based_scores(self, 
                                user_id: Union[int, str], 
                                item_features: pd.DataFrame) -> np.ndarray:
        """
        Get content-based scores for all items for a user.
        
        This is a simplified version - in practice, you'd want to use the user's
        interaction history to determine which item features they prefer.
        """
        # For simplicity, return a uniform distribution
        # In practice, you'd want to implement a proper content-based scoring
        return np.ones(len(self.item_ids)) / len(self.item_ids)


def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load sample data for demonstration.
    
    Returns:
        Tuple of (interactions, items, users) DataFrames
    """
    # Sample interactions
    interactions = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'item_id': [101, 102, 101, 103, 102, 104, 101, 105, 103, 106],
        'rating': [5, 4, 3, 5, 4, 2, 5, 3, 4, 5]
    })
    
    # Sample item features
    items = pd.DataFrame({
        'item_id': [101, 102, 103, 104, 105, 106],
        'title': [
            'The Shawshank Redemption', 
            'The Godfather', 
            'The Dark Knight',
            'Pulp Fiction',
            'The Lord of the Rings',
            'Forrest Gump'
        ],
        'genre': [
            'Drama', 
            'Crime,Drama', 
            'Action,Crime,Drama',
            'Crime,Drama',
            'Adventure,Fantasy',
            'Drama,Romance'
        ],
        'year': [1994, 1972, 2008, 1994, 2003, 1994],
        'rating': [9.3, 9.2, 9.0, 8.9, 8.9, 8.8]
    }).set_index('item_id')
    
    # Sample user features
    users = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'age': [25, 30, 35, 40, 45],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'preferred_genre': ['Drama', 'Crime', 'Action', 'Drama', 'Adventure']
    }).set_index('user_id')
    
    return interactions, items, users


if __name__ == "__main__":
    # Load sample data
    interactions, items, users = load_sample_data()
    
    # Initialize and fit the recommender
    recommender = MultiModalRecommender(n_recommendations=3)
    recommender.fit(interactions, items, users)
    
    # Generate recommendations for each user
    for user_id in users.index:
        recommendations = recommender.recommend(user_id)
        print(f"\nRecommendations for user {user_id}:")
        for item_id, score in recommendations:
            item_title = items.loc[item_id, 'title']
            print(f"- {item_title} (score: {score:.2f})")
