from fastapi import FastAPI
import torch
import pickle
import numpy as np
from model_arch import HybridNCF
from schemas import UserRequest, RecommendationResponse

app = FastAPI()

try:
    with open('mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    
    user_encoder = mappings['user_encoder']
    item_encoder = mappings['item_encoder']
    item_features_df = mappings['item_idx_to_features']

    device = torch.device('cpu')
    model = HybridNCF(
        num_users=len(mappings['user_encoder'].classes_),
        num_items=len(mappings['item_encoder'].classes_),
        num_cats=len(mappings['cat_encoder'].classes_),
        num_brands=len(mappings['brand_encoder'].classes_)
    )
    
    model.load_state_dict(torch.load('best_hybrid_model.pth', map_location=device))
    model.eval()

except Exception:
    pass

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(req: UserRequest):
    if req.user_id not in user_encoder.classes_:
        return {
            "status": "cold_start",
            "user_id": req.user_id,
            "products": []
        }

    user_idx = user_encoder.transform([req.user_id])[0]
    
    all_items = torch.tensor(item_features_df['item_idx'].values, dtype=torch.long)
    all_cats = torch.tensor(item_features_df['cat_idx'].values, dtype=torch.long)
    all_brands = torch.tensor(item_features_df['brand_idx'].values, dtype=torch.long)
    user_tensor = torch.tensor([user_idx] * len(all_items), dtype=torch.long)

    with torch.no_grad():
        scores = model(user_tensor, all_items, all_cats, all_brands)
    
    top_indices = torch.topk(scores, 10).indices.numpy()
    rec_ids = item_encoder.inverse_transform(all_items[top_indices].numpy())
    
    return {
        "status": "success",
        "user_id": req.user_id,
        "products": rec_ids.tolist()
    }