from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from google.genai import Client

# Charger les variables d'environnement
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY non trouvée. Vérifie ton fichier .env")

# Créer le client Gemini
client = Client(api_key=api_key)

app = FastAPI()

# CORS pour React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle de requête
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        system_instruction = (
            "Tu es un assistant virtuel spécialisé dans la santé mentale. "
            "Si l'utilisateur dit bonjour ou bonsoir, tu dois lui répondre "
            "poliment, lui souhaiter la bienvenue et l'inviter à poser ses questions. "
            "Tu ne poses jamais de diagnostic médical. "
            "Tu encourages des pratiques saines et bienveillantes."
        )

        prompt = f"""
{system_instruction}

Utilisateur :
{request.message}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return {"reply": response.text}

    except Exception as e:
        return {"error": str(e)}
