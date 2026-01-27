from pydantic import BaseModel
from typing import List

class UserRequest(BaseModel):
    user_id: str

class RecommendationResponse(BaseModel):
    status: str
    user_id: str
    products: List[str]