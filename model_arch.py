import torch
import torch.nn as nn

class HybridNCF(nn.Module):
    def __init__(self, num_users, num_items, num_cats, num_brands, embedding_dim=64):
        super(HybridNCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.cat_embedding = nn.Embedding(num_cats, embedding_dim // 2)
        self.brand_embedding = nn.Embedding(num_brands, embedding_dim // 2)

        concat_dim = embedding_dim + embedding_dim + (embedding_dim // 2) + (embedding_dim // 2)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_idx, item_idx, cat_idx, brand_idx):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        cat_emb = self.cat_embedding(cat_idx)
        brand_emb = self.brand_embedding(brand_idx)
        
        combined = torch.cat([user_emb, item_emb, cat_emb, brand_emb], dim=1)
        return self.sigmoid(self.fc_layers(combined)).squeeze()