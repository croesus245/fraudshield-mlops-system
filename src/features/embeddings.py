"""
Entity Embeddings for fraud detection.

WHY EMBEDDINGS?
- Categorical features (merchant_id, user_id) have thousands of unique values
- One-hot encoding = huge sparse matrices
- Embeddings = dense, learned representations that capture similarity

WHAT WE EMBED:
- Users: spending patterns, risk profile
- Merchants: fraud rate, category
- Devices: usage patterns, risk

HOW:
1. Train embeddings on historical transaction graph
2. Store in feature store
3. Look up at inference time
"""

import numpy as np
from typing import Optional
from pathlib import Path
import pickle
from dataclasses import dataclass
from loguru import logger

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class EmbeddingConfig:
    """Configuration for entity embeddings."""
    user_dim: int = 32
    merchant_dim: int = 16
    device_dim: int = 16
    min_frequency: int = 5  # Minimum occurrences to get embedding


class EntityEmbeddings:
    """
    Learn and store entity embeddings from transaction data.
    
    Uses co-occurrence patterns:
    - Users who shop at similar merchants get similar embeddings
    - Merchants with similar customer bases get similar embeddings
    
    This is a simplified approach. Production systems might use:
    - Graph neural networks
    - Contrastive learning
    - Pre-trained transaction embeddings
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        
        # Encoders
        self._user_encoder: Optional[LabelEncoder] = None
        self._merchant_encoder: Optional[LabelEncoder] = None
        self._device_encoder: Optional[LabelEncoder] = None
        
        # Embeddings
        self._user_embeddings: Optional[np.ndarray] = None
        self._merchant_embeddings: Optional[np.ndarray] = None
        self._device_embeddings: Optional[np.ndarray] = None
        
        # Default embeddings for unknown entities
        self._user_default: Optional[np.ndarray] = None
        self._merchant_default: Optional[np.ndarray] = None
        self._device_default: Optional[np.ndarray] = None
        
        self._is_fitted = False
    
    def fit(self, transactions: "pd.DataFrame") -> "EntityEmbeddings":
        """
        Learn embeddings from transaction data.
        
        Args:
            transactions: DataFrame with user_id, merchant_id, device_id columns
        """
        import pandas as pd
        
        logger.info(f"Fitting embeddings on {len(transactions)} transactions")
        
        # Filter to frequent entities
        user_counts = transactions["user_id"].value_counts()
        merchant_counts = transactions["merchant_id"].value_counts()
        device_counts = transactions["device_id"].value_counts()
        
        frequent_users = set(user_counts[user_counts >= self.config.min_frequency].index)
        frequent_merchants = set(merchant_counts[merchant_counts >= self.config.min_frequency].index)
        frequent_devices = set(device_counts[device_counts >= self.config.min_frequency].index)
        
        logger.info(f"Frequent entities: {len(frequent_users)} users, {len(frequent_merchants)} merchants, {len(frequent_devices)} devices")
        
        # Create encoders
        self._user_encoder = LabelEncoder()
        self._merchant_encoder = LabelEncoder()
        self._device_encoder = LabelEncoder()
        
        self._user_encoder.fit(list(frequent_users))
        self._merchant_encoder.fit(list(frequent_merchants))
        self._device_encoder.fit(list(frequent_devices))
        
        # Build co-occurrence matrices
        # User-Merchant matrix
        filtered = transactions[
            transactions["user_id"].isin(frequent_users) &
            transactions["merchant_id"].isin(frequent_merchants)
        ]
        
        user_indices = self._user_encoder.transform(filtered["user_id"])
        merchant_indices = self._merchant_encoder.transform(filtered["merchant_id"])
        
        n_users = len(self._user_encoder.classes_)
        n_merchants = len(self._merchant_encoder.classes_)
        
        # Co-occurrence matrix (sparse would be better for large data)
        user_merchant_matrix = np.zeros((n_users, n_merchants))
        for u, m in zip(user_indices, merchant_indices):
            user_merchant_matrix[u, m] += 1
        
        # Apply log transform (reduces impact of heavy users)
        user_merchant_matrix = np.log1p(user_merchant_matrix)
        
        # SVD for dimensionality reduction
        # User embeddings from rows, merchant embeddings from columns
        user_svd = TruncatedSVD(n_components=min(self.config.user_dim, n_merchants - 1))
        self._user_embeddings = user_svd.fit_transform(user_merchant_matrix)
        
        merchant_svd = TruncatedSVD(n_components=min(self.config.merchant_dim, n_users - 1))
        self._merchant_embeddings = merchant_svd.fit_transform(user_merchant_matrix.T)
        
        # Device embeddings (simpler: based on user co-occurrence)
        device_filtered = transactions[
            transactions["device_id"].isin(frequent_devices) &
            transactions["user_id"].isin(frequent_users)
        ]
        
        if len(device_filtered) > 0:
            device_indices = self._device_encoder.transform(device_filtered["device_id"])
            device_user_indices = self._user_encoder.transform(device_filtered["user_id"])
            
            n_devices = len(self._device_encoder.classes_)
            device_user_matrix = np.zeros((n_devices, n_users))
            for d, u in zip(device_indices, device_user_indices):
                device_user_matrix[d, u] += 1
            
            device_user_matrix = np.log1p(device_user_matrix)
            
            device_svd = TruncatedSVD(n_components=min(self.config.device_dim, n_users - 1))
            self._device_embeddings = device_svd.fit_transform(device_user_matrix)
        else:
            self._device_embeddings = np.zeros((len(self._device_encoder.classes_), self.config.device_dim))
        
        # Set default embeddings (mean of all)
        self._user_default = self._user_embeddings.mean(axis=0)
        self._merchant_default = self._merchant_embeddings.mean(axis=0)
        self._device_default = self._device_embeddings.mean(axis=0)
        
        self._is_fitted = True
        logger.info(f"Embeddings fitted: user={self._user_embeddings.shape}, merchant={self._merchant_embeddings.shape}, device={self._device_embeddings.shape}")
        
        return self
    
    def get_user_embedding(self, user_id: str) -> np.ndarray:
        """Get embedding for a user."""
        if not self._is_fitted:
            raise RuntimeError("Embeddings not fitted. Call fit() first.")
        
        try:
            idx = self._user_encoder.transform([user_id])[0]
            return self._user_embeddings[idx]
        except (ValueError, KeyError):
            return self._user_default
    
    def get_merchant_embedding(self, merchant_id: str) -> np.ndarray:
        """Get embedding for a merchant."""
        if not self._is_fitted:
            raise RuntimeError("Embeddings not fitted. Call fit() first.")
        
        try:
            idx = self._merchant_encoder.transform([merchant_id])[0]
            return self._merchant_embeddings[idx]
        except (ValueError, KeyError):
            return self._merchant_default
    
    def get_device_embedding(self, device_id: str) -> np.ndarray:
        """Get embedding for a device."""
        if not self._is_fitted:
            raise RuntimeError("Embeddings not fitted. Call fit() first.")
        
        try:
            idx = self._device_encoder.transform([device_id])[0]
            return self._device_embeddings[idx]
        except (ValueError, KeyError):
            return self._device_default
    
    def get_all_embeddings(self, user_id: str, merchant_id: str, device_id: str) -> dict[str, np.ndarray]:
        """Get all embeddings for a transaction."""
        return {
            "user_embedding": self.get_user_embedding(user_id),
            "merchant_embedding": self.get_merchant_embedding(merchant_id),
            "device_embedding": self.get_device_embedding(device_id),
        }
    
    def get_embedding_features(self, user_id: str, merchant_id: str, device_id: str) -> dict[str, float]:
        """
        Get embeddings as flat feature dict.
        
        Returns dict like {"user_emb_0": 0.5, "user_emb_1": -0.3, ...}
        """
        features = {}
        
        user_emb = self.get_user_embedding(user_id)
        for i, v in enumerate(user_emb):
            features[f"user_emb_{i}"] = float(v)
        
        merchant_emb = self.get_merchant_embedding(merchant_id)
        for i, v in enumerate(merchant_emb):
            features[f"merchant_emb_{i}"] = float(v)
        
        device_emb = self.get_device_embedding(device_id)
        for i, v in enumerate(device_emb):
            features[f"device_emb_{i}"] = float(v)
        
        return features
    
    def save(self, path: str) -> None:
        """Save embeddings to disk."""
        path = Path(path)
        
        data = {
            "config": self.config,
            "user_encoder": self._user_encoder,
            "merchant_encoder": self._merchant_encoder,
            "device_encoder": self._device_encoder,
            "user_embeddings": self._user_embeddings,
            "merchant_embeddings": self._merchant_embeddings,
            "device_embeddings": self._device_embeddings,
            "user_default": self._user_default,
            "merchant_default": self._merchant_default,
            "device_default": self._device_default,
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Embeddings saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "EntityEmbeddings":
        """Load embeddings from disk."""
        path = Path(path)
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        obj = cls(config=data["config"])
        obj._user_encoder = data["user_encoder"]
        obj._merchant_encoder = data["merchant_encoder"]
        obj._device_encoder = data["device_encoder"]
        obj._user_embeddings = data["user_embeddings"]
        obj._merchant_embeddings = data["merchant_embeddings"]
        obj._device_embeddings = data["device_embeddings"]
        obj._user_default = data["user_default"]
        obj._merchant_default = data["merchant_default"]
        obj._device_default = data["device_default"]
        obj._is_fitted = True
        
        logger.info(f"Embeddings loaded from {path}")
        return obj


class EmbeddingFeatureTransformer:
    """
    Feature transformer that adds entity embeddings.
    
    Use in the feature pipeline to add embedding features.
    """
    
    def __init__(self, embeddings: EntityEmbeddings):
        self.embeddings = embeddings
    
    def transform(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Add embedding features to dataframe."""
        import pandas as pd
        
        embedding_features = []
        
        for _, row in df.iterrows():
            features = self.embeddings.get_embedding_features(
                row["user_id"],
                row["merchant_id"],
                row.get("device_id", "unknown"),
            )
            embedding_features.append(features)
        
        embedding_df = pd.DataFrame(embedding_features)
        return pd.concat([df.reset_index(drop=True), embedding_df], axis=1)
