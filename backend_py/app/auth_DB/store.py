import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import jwt
from motor.motor_asyncio import AsyncIOMotorClient
import certifi

router = APIRouter(tags=["Store Usage Logs"])

# --- Security & JWT Decoding ---
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your_super_secret_jwt_key_here")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user_email(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the JWT token to get the user's email (stored as 'sub')
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return email
    except jwt.PyJWTError:
        raise credentials_exception

# --- Database Connection ---
mongo_client = None

def get_store_collection():
    global mongo_client
    if mongo_client is None:
        mongo_url = os.environ.get("MONGO_URL", "mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority")
        mongo_client = AsyncIOMotorClient(mongo_url, tlsCAFile=certifi.where())
    db = mongo_client["nexus"]
    return db["usage_logs"]  # Table/Collection dedicated strictly to history/usage logs

# --- Pydantic Models ---
class UsageLogCreate(BaseModel):
    api_name: str  # e.g., "journal_recommendation", "topic_selection", "literature_review"
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None  # Optionally store results too

# --- Routes ---

@router.post("/usage", status_code=status.HTTP_201_CREATED)
async def store_usage_log(
    log_data: UsageLogCreate, 
    email: str = Depends(get_current_user_email),
    collection = Depends(get_store_collection)
):
    """
    Stores an interaction/result from any of the APIs tied to the logged-in user.
    """
    log_document = {
        "user_id": email, # Using email to tie the log exactly to the authenticated user
        "api_name": log_data.api_name,
        "request_data": log_data.request_data,
        "response_data": log_data.response_data,
        "timestamp": datetime.now(timezone.utc)
    }
    
    result = await collection.insert_one(log_document)
    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Failed to store usage log")
        
    return {"message": "Usage log stored successfully", "log_id": str(result.inserted_id)}


@router.get("/usage", response_model=List[Dict[str, Any]])
async def get_user_usage_logs(
    api_name: Optional[str] = None,
    email: str = Depends(get_current_user_email),
    collection = Depends(get_store_collection)
):
    """
    Retrieves the history/logs for the logged-in user. 
    Can be optionally filtered by api_name.
    """
    query = {"user_id": email}
    if api_name:
        query["api_name"] = api_name
        
    # Sort them so the newest logs appear first
    cursor = collection.find(query).sort("timestamp", -1)
    
    logs = []
    async for document in cursor:
        document["_id"] = str(document["_id"])  # Convert MongoDB ObjectId to string for JSON serialization
        logs.append(document)
        
    return logs
