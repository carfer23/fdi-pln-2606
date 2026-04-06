"""Módulo de configuración para definir variables constantes."""

import os
from dotenv import load_dotenv

load_dotenv()

# Modelo de spaCy para procesamiento de texto
SPACY_MODEL = os.getenv("SPACY_MODEL")

# Modelo de Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Archivo de cache para embeddings
EMBEDDINGS_CACHE_FILE = os.getenv("CACHE_FILE")
