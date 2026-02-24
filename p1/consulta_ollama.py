"""Módulo que maneja la interacción con Ollama."""

import ollama
import re
import json

from config import OLLAMA_MODEL

def cargar_prompt(nombre_archivo, **kwargs):
    """Carga un prompt."""
    with open(f"prompts/{nombre_archivo}.txt", "r", encoding="utf-8") as f:
        plantilla = f.read()
    return plantilla.format(**kwargs)

def clean_json_response(response_text):
    """Limpia la respuesta del LLM para extraer solo el JSON válido."""

    # Eliminar bloques de código markdown
    response_text = re.sub(r'```json', '', response_text)
    response_text = re.sub(r'```', '', response_text)
    
    # Buscar el primer '{' y el último '}'
    start = response_text.find('{')
    end = response_text.rfind('}') + 1
    
    if start != -1 and end != -1:
        return response_text[start:end]
    return response_text

def ollama_generate(prompt: str, system_prompt: str = "") -> dict:
    """Consulta a Ollama y fuerza el retorno de un diccionario."""

    print("🤖 Pensando...")
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            #format='json',
            options={'temperature': 0.1} # Temperatura baja para ser más preciso con JSON
        )
        content = response['message']['content']
        cleaned_json = clean_json_response(content)
        
        print(cleaned_json)

        return json.loads(cleaned_json)
    
    except json.JSONDecodeError:
        print("⚠️ Error: El modelo no devolvió un JSON válido.")
        print(f"Respuesta cruda: {response['message']['content']}")
        return None
    
    except Exception as e:
        print(f"⚠️ Error Ollama: {e}")
        return None