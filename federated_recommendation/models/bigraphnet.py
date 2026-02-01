"""
Phase 5: BiGraphNet - Graph Neural Network for Recommendations

WHY: After getting embeddings from multi-modal model, we need to understand 
     relationships between users and items (posts). A graph structure is 
     perfect for this - users and items are nodes, interactions are edges.

WHAT: BiGraphNet (Bipartite Graph Network) learns from the graph structure:
      - Users connect to items they interacted with
      - Message passing: users learn from their items, items learn from their users
      - Better embeddings that capture relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class BipartiteGraphLayer(nn.Module):
    """
    Single layer of bipartite graph neural network
    
    In a bipartite graph:
    - User nodes only connect to item nodes
    - Item nodes only connect to user nodes
    - No user-user or item-item connections
    """
    def __init__(self, in_dim: int, out_dim: int):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
        """
        super(BipartiteGraphLayer, self).__init__()

        #transform for user nodes
        self.user_transform = nn.Linear(in_dim, out_dim)

        #transform for item nodes
        self.item_transform = nn.Linear(in_dim, out_dim)

        #activation
        self.activation = nn.ReLU() #helps for learn complex pattern

        #normalization
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        user_item_edge_index: torch.Tensor,
        item_user_edge_index: torch.Tensor
    )  -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with message passing
        
        Args:
            user_features: [num_users, in_dim]
            item_features: [num_items, in_dim]
            user_item_edge_index: [2, num_edges] edges from users to items
            item_user_edge_index: [2, num_edges] edges from items to users
            
        Returns:
           Updated user_features, item_features
        """   
        # message passing : user -> items
        # each item aggregates message from connected users ? why i need to aggregate the message
        item_messages = self._aggregate_messages(
            user_features,
            user_item_edge_index,
            item_features.shape[0]
        )
        user_messages = self.aggregate_messages(
           item_features,
           item_user_edge_index,
           user_features.shape[0]
        )

        #transform and combine
        user_features_new = self.user_transform(user_features + user_messages)
        item_features_new = self.item_transform(item_features, item_messages)

        # apply activation and normalization
        user_features_new = self.norm(self.activation(user_features_new))
        item_features_new = self.norm(self.activation(item_features_new))

        return user_features_new, item_features_new
    
    def _aggregate_message(
        self,
        source_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_target_nodes: int
    ) -> torch.Tensor:
        """
        Aggregate messages from source nodes to target nodes
        
        Args:
            source_features: Features of source nodes [num_source, dim]
            edge_index: [2, num_edges] - [source_idx, target_idx]
            num_target_nodes: Number of target nodes
            
        Returns:
            Aggregated messages for target nodes [num_target, dim]
        """

        # get the source and traget indices
        source_idx = edge_index[0]
        target_idx = edge_index[1]

        # get the features of source nodes involved in edges
        messages = source_features[source_idx]
        
        # aggregate messages for each target node (mean aggregation)
        aggregated  = torch.zeros(
            num_target_nodes,
            source_features.shape[1],
            device = source_features.device
        )

        # sum messages for each target 
        aggregated.index_add_(0, target_idx, messages)

        # count how many messages each target received
        degree = torch.zeros(num_target_nodes, device=source_features.device)
        degree.index_add_(0, target_idx, torch.ones_like(target_idx, dtype=torch.float))

        # average ( avoid division by zero )
        degree = torch.clamp(degree, min=1.0).unsqueeze(1)
        aggregated = aggregated / degree

        return aggregated

class BiGraphNet(nn.Module):
    """
    - Initial embeddings for users and items (from multi-modal model)
    - Multiple BipartiteGraphLayer for message passing
    - Final embeddings used for recommendation
    """     

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dims: list = [256, 128, 64],
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        """
        Args:
            embedding_dim: Initial embedding dimension
            hidden_dims: Hidden dimensions for each layer
            num_layers: Number of graph layers
            dropout: Dropout probability
        """
        super(BiGraphNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # graph layers
        dims = [embedding_dim] + hidden_dims[:num_layers]
        self.graph_layers = nn.ModuleList([
            BipartiteGraphLayer(dims[i], dims[i+1])
            for i in range(num_layers)
        ])

        # dropout
        self.dropout = nn.Dropout(dropout)

        # output dimension 
        self.output_dim = dims[-1]


    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        user_item_edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the graph network
        
        Args:
            user_features: Initial user embeddings [num_users, embedding_dim]
            item_features: Initial item embeddings [num_items, embedding_dim]
            user_item_edges: Edge list [2, num_edges]
                            First row: user indices
                            Second row: item indices
            
        Returns:
            Final user embeddings, final item embeddings
        """
        #create reverse edges(items to user)
        item_user_edges = torch.stack([user_item_edges[1], user_item_edges[0]], dim=0) 

        # messages passing through lalyers
        for layer in self.graph_layers:
            user_features, item_features = layer(
                user_features,
                item_features,
                user_item_edges,
                item_user_edges
            )

            #apply dropout (not in last layer )
            user_features = self.dropout(user_features)
            item_features = self.dropout(item_features)

        return user_features, item_features

    def get_output_dim(self) -> int:
        "Return output embedding dimension"
        return self.output_dim  

class RecommendationSystem(nn.Module):
    """
    Complete recommendation system combining multi-modal model and BiGraphNet
    """

    def __init__(
        self,
        multimodal_model,
        num_users: int,
        num_items: int,
        embedding_dim: int = 256,
        graph_hidden_dims: list = [256, 128, 64]    
    ):
        """
        Args:
            multimodal_model: Trained multi-modal model (Phase 4)
            num_users: Number of users in the system
            num_items: Number of items (posts)
            embedding_dim: Embedding dimension
            graph_hidden_dims: Hidden dimensions for graph layers
        """
        super(RecommendationSystem, self).__init__()

        self.multimodal_model = multimodal_model
        self.num_users = num_users
        self.num_items = num_items

        # freeze multi-modal model (use it as feature extraction )
        for param in self.multimodal_model.parameters():
            param.requires_grad = False

        # user embeddings ( learnable , initialized randomly)
        self.user_embeddngs = nn.Embedding(num_users, embedding_dim)
        nn.init.xavier_uniform_(self.user_embeddngs.weight)

        # bigraphNet
        self.bigraphnet = BiGraphNet(
            embedding_dim=embedding_dim,
            hidden_dims=graph_hidden_dims,
            num_layers=3,
            dropout=0.3
        )   

    def get_item_embeddings(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        images: torch.Tensor    
    )   -> torch.Tensor:
        """
        Get item embeddings from multi-modal model
        
        Args:
            text_input_ids: Text token IDs [num_items, seq_len]
            text_attention_mask: Attention masks [num_items, seq_len]
            images: Images [num_items, 3, 224, 224]
            
        Returns:
            Item embeddings [num_items, embedding_dim]
        """ 
        with torch.no_grad():
            embeddings = self.multimodal_model.get_embeddings(
                text_input_ids,
                text_attention_mask,
                images
            )
        return embeddings
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_text_ids: torch.Tensor,
        item_attention_masks: torch.Tensor,
        item_images: torch.Tensor,
        user_item_edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            user_ids: User IDs [num_users]
            item_text_ids: Item text IDs [num_items, seq_len]
            item_attention_masks: Item attention masks [num_items, seq_len]
            item_images: Item images [num_items, 3, 224, 224]
            user_item_edges: Interaction edges [2, num_edges]
            
        Returns:
            User embeddings, Item embeddings after graph processing
        """

        # get initial embeddings
        user_features = self.user_embeddngs(user_ids) # [num_items, 256]
        item_features = self.get_item_embeddings(
            item_text_ids,
            item_attention_masks,
            item_images
        ) # [num_items, 256]

        # pass through bigraphNet
        user_features, item_features = self.bigraphnet(
            user_features,
            item_features,
            user_item_edges
        )

        return user_features, item_features
    
    def recommend(
        self,
        user_id: int,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        k: int = 10,
        exclude_items: Optional[list] = None    
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate top-K recommendations for a user
        
        Args:
            user_id: User ID
            user_embeddings: All user embeddings [num_users, dim]
            item_embeddings: All item embeddings [num_items, dim]
            k: Number of recommendations
            exclude_items: Items to exclude (already interacted)
            
        Returns:
            Top-K item indices, Top-K scores
        """
        #get user embedding
        user_emb = user_embeddings[user_id].unsqueeze(0)

        # calculate similarity with all items
        similarities = F.cosine_similarity(
            user_emb,
            item_embeddings,
            dim=1
        ) # [num_items]

        # exclude already interacted items
        if exclude_items is not None:
            similarities[exclude_items] = -float('inf')

        # get top-k
        top_k_scores, top_k_indices = torch.topk(similarities, k)

        return top_k_indices, top_k_scores

# ========================
# UTILITY FUNCTIONS
# ========================

def create_interaction_graph(interations: list) -> torch.Tensor:
    """
    Create edge list from interactions
    
    Args:
        interactions: List of (user_id, item_id) tuples
        
    Returns:
        Edge tensor [2, num_edges]
    """   
    if len(interations) == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    
    user_ids = [i[0] for i in interations]
    item_ids = [i[0] for i in interations]

    edge_index = torch.Tensor([user_ids, item_ids], dtype=torch.long)

    return edge_index

def generate_systhetic_interactions(
        num_users: int,
        num_items: int,
        num_interactions: int
) -> torch.Tensor:
    """
    Generate synthetic user-item interactions for testing
    
    Args:
        num_users: Number of users
        num_items: Number of items
        num_interactions: Number of interactions to generate
        
    Returns:
        Edge tensor [2, num_interactions]
    """
    user_ids = torch.randint(0, num_users, (num_interactions,))
    item_ids = torch.randint(0, num_items, (num_interactions,))

    edge_index = torch.stack([user_ids, item_ids], dim=0)
    return edge_index
        
# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing BiGraphNet - Phase 5")
    print("="*70)
    
    # Configuration
    num_users = 100
    num_items = 500
    num_interactions = 2000
    embedding_dim = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create synthetic data
    print(f"\nCreating synthetic data:")
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Interactions: {num_interactions}")
    
    # Generate interactions (user-item edges)
    edges = generate_systhetic_interactions(num_users, num_items, num_interactions)
    print(f"  Edge shape: {edges.shape}")
    
    # Create initial embeddings (simulating multi-modal output)
    user_features = torch.randn(num_users, embedding_dim)
    item_features = torch.randn(num_items, embedding_dim)
    
    print(f"\nInitial embeddings:")
    print(f"  User features: {user_features.shape}")
    print(f"  Item features: {item_features.shape}")
    
    # Create BiGraphNet
    print(f"\nCreating BiGraphNet...")
    bigraphnet = BiGraphNet(
        embedding_dim=embedding_dim,
        hidden_dims=[256, 128, 64],
        num_layers=3,
        dropout=0.3
    )
    bigraphnet = bigraphnet.to(device)
    
    # Move data to device
    user_features = user_features.to(device)
    item_features = item_features.to(device)
    edges = edges.to(device)
    
    # Forward pass
    print(f"\nRunning forward pass...")
    with torch.no_grad():
        user_emb_out, item_emb_out = bigraphnet(
            user_features,
            item_features,
            edges
        )
    
    print(f"Output embeddings:")
    print(f"  User: {user_emb_out.shape}")
    print(f"  Item: {item_emb_out.shape}")
    
    # Test recommendation
    print(f"\nGenerating recommendations for user 0...")
    user_id = 0
    top_k_items, top_k_scores = bigraphnet.recommend(
        user_id=user_id,
        user_embeddings=user_emb_out,
        item_embeddings=item_emb_out,
        k=10
    )
    
    print(f"Top 10 recommended items: {top_k_items.cpu().numpy()}")
    print(f"Scores: {top_k_scores.cpu().numpy()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in bigraphnet.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n" + "="*70)
    print("BiGraphNet Test Complete! ✓")
    print("="*70)


            
