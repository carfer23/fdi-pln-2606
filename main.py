import requests
import json
import subprocess
import sys
import time
from datetime import datetime

URL_BASE = "http://147.96.81.252:8000"
OLLAMA_BIN = "/home/hlocal/Documents/ollama/bin/ollama"
AGENT_NAME = "trilobite"
OLLAMA_MODEL = "qwen3-vl:8b"

OLLAMA_URL = "http://127.0.0.1:42661"

def register_agent(name: str):
    """Registra un alias."""
    url = f"{URL_BASE}/alias/{name}"
    r = requests.post(url)

    if r.status_code == 200:
        print(f"Alias '{name}' registrado correctamente")
    else:
        print("Error al registrar el alias")
        print("Status:", r.status_code)
        print("Respuesta:", r.text)
        sys.exit(1)

def gente():
    """Mira los alias registrados."""
    url = f"{URL_BASE}/gente"
    r = requests.get(url)

    if r.status_code == 200:
        return r.text
    return []
    
def get_recursos():
    url = f"{URL_BASE}/info"
    r = requests.get(url)

    if r.status_code == 200:
        r = json.loads(r.text)
        print(f"Recursos: {r["Recursos"]}")

def get_objetivo():
    url = f"{URL_BASE}/info"
    r = requests.get(url)

    if r.status_code == 200:
        r = json.loads(r.text)
        print(f"Objetivo: {r["Objetivo"]}")

def get_buzon():
    url = f"{URL_BASE}/info"
    r = requests.get(url)

    if r.status_code == 200:
        r = json.loads(r.text)
        print(f"Buzon: {r["Buzon"]}")

def enviar_carta(dest, asunto, cuerpo):
    url = f"{URL_BASE}/carta"
    data = {
        "remi": AGENT_NAME,
        "dest": dest,
        "asunto": asunto,
        "cuerpo": cuerpo
        }
    r = requests.post(url, data)

    if r.status_code == 200:
        print("Carta enviada correctamente")

def enviar_paquete(destinatario, objeto, cantidad):
    url = f"{URL_BASE}/paquete/{destinatario}"
    data = {
        objeto: cantidad
        }
    r = requests.post(url, data)

    if r.status_code == 200:
        print("Paquete enviado correctamente")

def ollama_generate(prompt: str) -> str:
    """
    Llama al endpoint /api/generate del servidor Ollama
    """
    #url = f"{OLLAMA_URL}/api/generate"
    url = "http://127.0.0.1:42661/api/chat"
    
    # payload = {
    #     "model": OLLAMA_MODEL,
    #     "prompt": prompt,
    #     "max_tokens": 512
    # }
    payload = {
    "messages": [
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": "Hola, ¿puedes explicarme la teoría de la relatividad?"}
        ]
    }

    r = requests.post(url, json=payload)

    if r.status_code == 200:
        data = r.json()
        # Depende de la respuesta de Ollama
        # Normalmente data["completion"] o data["results"][0]["text"]
        # Ajustamos según el formato real:
        if "completion" in data:
            return data["completion"]
        elif "results" in data and len(data["results"]) > 0:
            return data["results"][0]["text"]
        else:
            return str(data)
    else:
        return f"Error {r.status_code}: {r.text}"

def main():
    print("Iniciando agente IA (ruta absoluta)")

    # alias = gente()
    # lista_alias = json.loads(alias)
    # if AGENT_NAME not in lista_alias:
    #     register_agent(AGENT_NAME)
    # else:
    #     print("Alias ya registrado.")

    # get_recursos()

    # get_objetivo()

    response = ollama_generate("Eres un agente que maneja recursos y los intercambia con otros usuarios. " \
    "Debes enviar cartas para pedir los recursos objetivo (los que no tienes) y recibirás cartas de otros usuarios" \
    "pidiendo recursos que puedes tener o no.")

    print("\nRespuesta:\n")
    print(response)

    while True:

        texto = input()
        response = ollama_generate(texto)

        print("\nRespuesta:\n")
        print(response)


if __name__ == "__main__":
    main()

# Registro con Butler

# While True

    # Miramos que gente hay endpoint gente

    # Intento conseguir recursos
