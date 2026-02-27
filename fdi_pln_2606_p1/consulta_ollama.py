"""Módulo que maneja la interacción con Ollama."""

import ollama
import json
from pathlib import Path

from .config import OLLAMA_MODEL

P1_DIR = Path(__file__).resolve().parent


def cargar_prompt(nombre_archivo, **kwargs):
    """Carga un prompt."""
    with open(f"{P1_DIR}/prompts/{nombre_archivo}.txt", "r", encoding="utf-8") as f:
        plantilla = f.read()
    return plantilla.format(**kwargs)


def clean_json_response(response_text):
    """Limpia la respuesta del LLM para extraer solo el JSON válido."""

    texto = response_text.replace("```json", "").replace("```", "").strip()
    try:
        datos = json.loads(texto)
        return datos
    except json.JSONDecodeError:
        print("❌ Error: El agente falló al generar un JSON válido.")
        print(f"Respuesta cruda: {response_text}")
        # Retornar una acción 'esperar' por defecto para que el bot no crashee
        return {"accion": "esperar"}


def ollama_generate(prompt: str, system_prompt: str = "") -> dict:
    """Consulta a Ollama y fuerza el retorno de un diccionario."""

    print("🤖 Pensando...")
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            # format='json',
            # think=False,
            options={
                "temperature": 0.1
            },  # Temperatura baja para ser más preciso con JSON
        )
        content = response["message"]["content"]
        cleaned_json = clean_json_response(content)

        print(f"JSON de respuesta: {cleaned_json}")

        return cleaned_json

    except json.JSONDecodeError:
        print("⚠️ Error: El modelo no devolvió un JSON válido.")
        print(f"Respuesta cruda: {response['message']['content']}")
        return None

    except Exception as e:
        print(f"⚠️ Error Ollama: {e}")
        return None
