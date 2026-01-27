import requests
import json
import subprocess
import sys
import time

URL_BASE = "http://147.96.81.252:8000"
OLLAMA_BIN = "/home/hlocal/Documents/PLN/bin/ollama"
AGENT_NAME = "trilobite"
OLLAMA_MODEL = "qwen3-vl:8b"

def register_agent(name: str):
    url = f"{URL_BASE}/alias/{name}"
    r = requests.post(url)

    if r.status_code == 200:
        print(f"Alias '{name}' registrado correctamente")
    else:
        print("Error al registrar el alias")
        print("Status:", r.status_code)
        print("Respuesta:", r.text)
        sys.exit(1)

def ollama_generate(prompt: str) -> str:
    try:
        result = subprocess.run(
            [OLLAMA_BIN, "run", OLLAMA_MODEL],
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout.strip()

    except FileNotFoundError:
        print("No se encontró el binario de Ollama")
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        print("Error ejecutando Ollama")
        print(e.stderr)
        sys.exit(1)

def main():
    print("Iniciando agente IA (ruta absoluta)")

    register_agent(AGENT_NAME)

    # response = ollama_generate(
    #     "Preséntate como un agente de IA que acaba de registrarse en el sistema."
    # )

    # print("\nRespuesta del agente:\n")
    # print(response)

    url_info = f"{URL_BASE}/info"
    r = requests.get(url_info)
    print(r.text)


if __name__ == "__main__":
    main()

# app = FastAPI()
# url_base = "http://147.96.81.252:8000"

# @app.post("/register/{nombre}")
# def add_alias(nombre: str):
#     url_alias = f"{url_base}/alias/{nombre}"
#     r = requests.post(url_alias)
#     if r.status_code == 200:
#         return {"message": f"Alias '{nombre}' registrado!"}
#     else:
#         return {
#             "message": "No se ha podido registrar",
#             "status": r.status_code,
#             "respuesta": r.text
#         }

# Registro con Butler

# While True

# Intento conseguir recursos
