"""Módulo de configuración para definir variables constantes."""

import os
from dotenv import load_dotenv

load_dotenv()


def _read_env(name: str) -> str | None:
    """Lee una variable de entorno y limpia espacios extra."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


# URL de Butler
URL_BASE = _read_env("FDI_PLN__BUTLER_ADDRESS")

# Nombre del agente
AGENT_NAME = _read_env("AGENT_NAME")

# Modelo de Ollama
OLLAMA_MODEL = _read_env("OLLAMA_MODEL")
