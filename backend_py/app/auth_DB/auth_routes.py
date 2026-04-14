import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import jwt
from motor.motor_asyncio import AsyncIOMotorClient
import certifi

# --- Router Setup ---
router = APIRouter(prefix="/auth", tags=["Authentication"])

# --- Security Configuration ---
# You can set these in your .env file
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your_super_secret_jwt_key_here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day expiration

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Database Connection ---
mongo_client = None

def get_user_collection():
    global mongo_client
    if mongo_client is None:
        # User will enter MONGO_URL in environment variables later
        mongo_url = os.environ.get("MONGO_URL", "mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority")
        mongo_client = AsyncIOMotorClient(mongo_url, tlsCAFile=certifi.where())
    
    # Returning the 'users' collection in the 'nexus' database
    db = mongo_client["nexus"]
    return db["users"]

# --- Pydantic Models ---

class UserSignup(BaseModel):
    # Personal Info
    full_name: str
    email: EmailStr
    password: str
    role: str
    
    # Academic/Professional Info
    institution: str
    department: str
    designation: str
    education_level: str
    
    # Preferences
    preferred_language: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# --- Helper Functions ---

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- API Routes ---

@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(user: UserSignup, collection = Depends(get_user_collection)):
    # 1. Check if user already exists
    existing_user = await collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists."
        )
    
    # 2. Hash the user's password
    user_dict = user.model_dump()
    user_dict["password"] = get_password_hash(user_dict["password"])
    
    # 3. Add metadata
    user_dict["created_at"] = datetime.now(timezone.utc)
    
    # 4. Save to MongoDB
    result = await collection.insert_one(user_dict)
    if not result.inserted_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user."
        )
        
    # 5. Generate JWT Access Token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role}, 
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
async def login(user_credentials: UserLogin, collection = Depends(get_user_collection)):
    # 1. Check if user exists
    user_record = await collection.find_one({"email": user_credentials.email})
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # 2. Verify password against the hashed password structure in DB
    if not verify_password(user_credentials.password, user_record["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # 3. Generate JWT Access Token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_record["email"], "role": user_record.get("role", "student")}, 
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

@router.get("/me", status_code=status.HTTP_200_OK)
async def get_current_user_profile(token: str = Depends(oauth2_scheme), collection = Depends(get_user_collection)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_record = await collection.find_one({"email": email})
        if not user_record:
            raise HTTPException(status_code=404, detail="User not found")
            
        user_record.pop("password", None)
        user_record["_id"] = str(user_record["_id"])
        return user_record
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

