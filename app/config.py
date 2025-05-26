import os
from dotenv import load_dotenv

load_dotenv()

#Configurazione variabili d'ambiente
APP_ID = os.getenv("APP_ID")
APP_PASSWORD = os.getenv("APP_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

