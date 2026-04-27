"""Módulo que maneja la interacción con Ollama."""

import re
import json
import ollama
from pathlib import Path

if __package__:
    from .config import OLLAMA_MODEL
else:
    from config import OLLAMA_MODEL

P1_DIR = Path(__file__).resolve().parent


def cargar_prompt(nombre_archivo: str, **kwargs) -> str:
    """Carga y formatea un prompt desde archivo."""
    path = P1_DIR / "prompts" / f"{nombre_archivo}.txt"
    return path.read_text(encoding="utf-8").format(**kwargs)


def _extraer_json(text: str) -> dict:
    """Extrae un objeto JSON del texto de respuesta del LLM."""
    # Eliminar bloques de razonamiento interno de modelos tipo qwen3
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: buscar el primer objeto JSON completo en el texto
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    print(f"❌ No se pudo extraer JSON válido. Respuesta:\n{text[:400]}")
    return {"accion": "esperar"}


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
            format="json",
            options={"temperature": 0.1},
            think=False,
        )
        content = response["message"]["content"]
        result = _extraer_json(content)
        print(f"📋 Decisión: {result.get('accion')} → {result.get('razonamiento', '')[:120]}")
        return result

    except Exception as e:
        print(f"⚠️ Error Ollama: {e}")
        return {"accion": "esperar"}
