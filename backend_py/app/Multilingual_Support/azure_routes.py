from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests, uuid, os

router = APIRouter()

# 🔹 Request schema
class TranslateRequest(BaseModel):
    text: str
    to: str   # "te", "en", "ur", "hi"


# 🔹 Azure config
API_KEY = os.getenv("TRANSLATOR_TEXT_SUBSCRIPTION_KEY")
ENDPOINT = "https://api.cognitive.microsofttranslator.com"

# 🔹 Allowed languages
ALLOWED_LANGS = ["en", "te", "ur", "hi"]


@router.post("/translate-text")
async def translate_text(req: TranslateRequest):
    try:
        if req.to not in ALLOWED_LANGS:
            raise HTTPException(
                status_code=400,
                detail="Only 'en', 'te', 'ur', 'hi' supported"
            )

        url = ENDPOINT + f"/translate?api-version=3.0&to={req.to}"

        headers = {
            "Ocp-Apim-Subscription-Key": API_KEY,
            "Content-Type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4())
        }

        # 🔹 Clean text
        clean_text = " ".join(req.text.split())

        body = [{"text": clean_text}]

        res = requests.post(url, headers=headers, json=body)

        if res.status_code != 200:
            raise HTTPException(status_code=res.status_code, detail=res.text)

        data = res.json()
        translated = data[0]["translations"][0]["text"]

        return {
            "original": req.text,
            "translated": translated,
            "to": req.to
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))