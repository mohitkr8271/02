import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="Loan Advisor Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


class OTPRequest(BaseModel):
    email: str
    userId: str


class VerifyOTPRequest(BaseModel):
    userId: str
    otp: str


@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {"status": "ok", "message": "Backend service is running"}


@app.post("/api/send-otp")
async def send_otp(request: OTPRequest):
    """Proxy to Supabase edge function for sending OTP"""
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        
        response = supabase.functions.invoke(
            "send-otp-email",
            invoke_options={"body": {"email": request.email, "userId": request.userId}}
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify-otp")
async def verify_otp(request: VerifyOTPRequest):
    """Proxy to Supabase edge function for verifying OTP"""
    try:
        if not supabase:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        
        response = supabase.functions.invoke(
            "verify-otp",
            invoke_options={"body": {"userId": request.userId, "otp": request.otp}}
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
