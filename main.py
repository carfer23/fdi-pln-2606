import requests
import json
import sys
import ollama

URL_BASE = "http://147.96.81.252:7719"
OLLAMA_BIN = "/home/hlocal/Documents/ollama/bin/ollama"
AGENT_NAME = "halcón sagaz"
OLLAMA_MODEL = "qwen3-vl:4b"

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
        #print(f"Recursos: {r["Recursos"]}")
        return r["Recursos"]

def get_objetivo():
    url = f"{URL_BASE}/info"
    r = requests.get(url)

    if r.status_code == 200:
        r = json.loads(r.text)
        #print(f"Objetivo: {r["Objetivo"]}")
        return r["Objetivo"]

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
        print(f"Carta enviada correctamente a {dest}")
        print(cuerpo)
    else:
        print(f"Error al enviar carta a {dest}. Status: {r.status_code}, Respuesta: {r.text}")

def enviar_paquete(destinatario, objeto, cantidad):
    url = f"{URL_BASE}/paquete/{destinatario}"
    data = {
        objeto: cantidad
        }
    r = requests.post(url, data)

    if r.status_code == 200:
        print("Paquete enviado correctamente")

def ollama_generate(prompt: str, role: str) -> str:
    """
    Hace una consulta a Ollama.
    """
    print("Enviando prompt a Ollama...")
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {'role': role, 'content': prompt}
        ]
    )

    return response['message']['content']

def ejecutar_accion(respuesta_json: dict):
    accion = respuesta_json["accion"]

    if accion == "enviar_carta":
        generar_carta(
            respuesta_json["destinatario"],
            respuesta_json["recurso_necesitado"],
            respuesta_json["cantidad_recurso_necesitado"],
            respuesta_json["recurso_ofrecido"],
            respuesta_json["cantidad_recurso_ofrecido"]
        )

    elif accion == "enviar_paquete":
        enviar_paquete(
            respuesta_json["destinatario"],
            respuesta_json["objeto"],
            respuesta_json["cantidad"]
        )

    elif accion == "esperar":
        print("El agente decide esperar.")

def generar_carta(destinatario: str, recurso_necesitado: str, recurso_ofrecido: str, cantidad_recurso_necesitado, cantidad_recurso_ofrecido):
    """
    Genera la carta que el agente enviará a otro usuario solicitando intercambio de recursos.
    """
    asunto = f"Intercambio de recursos con {destinatario}"
    cuerpo = f"""
    Estimado usuario {destinatario},

    Soy el agente {AGENT_NAME} y me encuentro en busca de los siguientes recursos:

    - Recurso solicitado: {recurso_necesitado} - Cantidad: {cantidad_recurso_necesitado}
    - Recurso ofrecido: {recurso_ofrecido} - Cantidad: {cantidad_recurso_ofrecido}

    Ofrezco los recursos mencionados a cambio del recurso solicitado para poder acercarme a mi objetivo.

    Espero una respuesta favorable.

    Atentamente,
    {AGENT_NAME}
    """
    
    # Llamada a la función que envía la carta
    enviar_carta(destinatario, asunto, cuerpo)

def main():
    print("Iniciando agente IA...")

    alias = gente()
    lista_alias = json.loads(alias)
    if AGENT_NAME not in lista_alias:
        register_agent(AGENT_NAME)
    else:
        print("Alias ya registrado.")

    recursos = get_recursos()
    objetivo = get_objetivo()
    
    prompt_inicial = f"""
        Eres un agente que maneja recursos y los intercambia con otros usuarios.

        Tu objetivo es conseguir recolectar los recursos objetivo.
        Al inicio dispones de unos recursos determinados.

        Para conseguir los recursos objetivo:
        - debes enviar cartas para pedirlos
        - recibirás cartas de otros usuarios pidiendo recursos que puedes tener o no

        Para intercambiar recursos con otros usuarios debes enviar paquetes con esos recursos
        solo si con ello te acercas a tu objetivo.

        Tu usuario es: {AGENT_NAME}. No puedes enviarte cartas ni paquetes a ti mismo.

        Lo primero que tienes que hacer es enviar una carta a cada usuario solicitando los recursos que te faltan para cumplir
        el objetivo y los que puedes ofrecer. Por cada carta solo puede solicitar y ofrecer un recurso (la cantidad da igual).
        Para enviar una carta, en el JSON de respuesta indica la acción enviar_carta e indica el recurso a solicitar y su cantidad, 
        y el recurso a ofrecer y su cantidad.

        Recursos iniciales:
        {recursos}

        Recursos objetivo:
        {objetivo}

        Usuarios:
        {lista_alias}

        REGLAS DE RESPUESTA:
        - Responde SIEMPRE en JSON válido
        - No expliques nada fuera del JSON
        - Usa una de estas acciones:

        ACCIONES POSIBLES:
        1) enviar_carta
        2) enviar_paquete
        3) esperar

        FORMATO:

        {
        "accion": "...",
        "destinatario": "...",
        "objeto_enviar": "...",
        "cantidad_enviar": 0,
        "recurso_solicitado": "...",
        "cantidad_recurso_solicitado": 0,
        "recurso_ofrecido": "...",
        "cantidad_recurso_ofrecido": 0
        }
        """
    
    print(prompt_inicial)

    response = ollama_generate(prompt_inicial, "system")

    print("\nRespuesta:\n")
    print(response)

    while True:

        prompt = estado = {
        "recursos": get_recursos(),
        "objetivo": get_objetivo(),
        "buzon": get_buzon(),
        "usuarios": lista_alias
        }

        prompt = f"""
        ESTADO ACTUAL:
        {json.dumps(estado, indent=2)}

        Decide la siguiente acción.
        """

        response = ollama_generate(prompt, "user")

        try:
            accion = json.loads(response)
            print(f"Acción elegida por el agente: {accion}")
            ejecutar_accion(accion)
        except json.JSONDecodeError:
            print("Respuesta inválida del modelo:")
            print(response)


if __name__ == "__main__":
    main()

# Registro con Butler

# While True

    # Miramos que gente hay endpoint gente

    # Intento conseguir recursos
