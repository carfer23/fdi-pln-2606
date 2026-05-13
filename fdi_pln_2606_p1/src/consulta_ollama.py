"""Módulo que maneja la interacción con Ollama."""

import ollama
from pathlib import Path
from pydantic import BaseModel
from typing import Literal

if __package__:
    from .config import OLLAMA_MODEL
else:
    from config import OLLAMA_MODEL

P1_DIR = Path(__file__).resolve().parent


class DecisionAgente(BaseModel):
    razonamiento: str
    accion: Literal["enviar_paquete", "esperar"]
    destinatario: str = ""
    recurso_enviar: str = ""
    cantidad_recurso_enviar: int = 0
    recurso_recibir: str = ""
    cantidad_recurso_recibir: int = 0


def cargar_prompt(nombre_archivo: str, **kwargs) -> str:
    """Carga y formatea un prompt desde archivo."""
    path = P1_DIR / "prompts" / f"{nombre_archivo}.txt"
    return path.read_text(encoding="utf-8").format(**kwargs)


def ollama_generate(prompt: str, system_prompt: str = "") -> dict:
    """Consulta a Ollama y devuelve la decisión del agente como diccionario."""
    print("🤖 Pensando...")
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            format=DecisionAgente.model_json_schema(),
            options={"temperature": 0.1},
        )
        decision = DecisionAgente.model_validate_json(response["message"]["content"])
        result = decision.model_dump()
        print(f"📋 Decisión: {result.get('accion')} → {result.get('razonamiento', '')[:120]}")
        return result

    except Exception as e:
        print(f"⚠️ Error Ollama: {e}")
        return {"accion": "esperar"}
