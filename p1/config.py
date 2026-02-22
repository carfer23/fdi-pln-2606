"""Módulo de configuración para definir variables constantes."""

import os
from dotenv import load_dotenv

load_dotenv()

# URL de Butler
URL_BASE = os.getenv("FDI_PLN__BUTLER_ADDRESS")

# Nombre del agente
AGENT_NAME = os.getenv("AGENT_NAME")

# Modelo de Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")