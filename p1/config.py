"""Módulo de configuración para definir variables constantes."""

import os
from dotenv import load_dotenv

load_dotenv()

AGENT_NAME = "trilobite"
URL_BASE = os.getenv("FDI_PLN__BUTLER_ADDRESS")
#URL_BASE = "http://147.96.81.252:7719"
OLLAMA_MODEL = "qwen3-vl:4b"
OLLAMA_BIN = "/home/hlocal/Documents/ollama/bin/ollama"