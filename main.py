import requests
import json
import sys
import ollama
import time
import re

URL_BASE = "http://147.96.81.252:7719"
OLLAMA_BIN = "/home/hlocal/Documents/ollama/bin/ollama"
AGENT_NAME = "trilobite"
OLLAMA_MODEL = "qwen3-vl:4b"

def register_agent(name: str):
    """Registra un alias."""
    url = f"{URL_BASE}/alias/{name}"
    r = requests.post(url)

    if r.status_code == 200:
        print(f"Alias '{name}' registrado correctamente")
    else:
        print(f"Alias '{name}' ya estaba registrado")

def get_gente():
    """Mira los alias registrados."""
    url = f"{URL_BASE}/gente"
    r = requests.get(url)

    if r.status_code == 200:
        gente = json.loads(r.text)
    
        usuarios = []
        for user in gente:
            usuarios.append(user['alias'])

        return usuarios
    return []

def get_info():
    """Obtiene toda la info del agente de una vez"""
    try:
        r = requests.get(f"{URL_BASE}/info")
        if r.status_code == 200:
            return json.loads(r.text)
    except Exception as e:
        print(f"Error conectando: {e}")
    return None
    
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

def calcular_estado(info):
    """Calcula qué sobra y qué falta."""
    recursos = info.get("Recursos", {})
    objetivo = info.get("Objetivo", {})
    
    faltantes = {}
    sobrantes = {}
    
    # Calcular faltantes (Lo que necesito - Lo que tengo)
    for k, v in objetivo.items():
        actual = recursos.get(k, 0)
        diff = v - actual
        if diff > 0:
            faltantes[k] = diff

    # Calcular sobrantes (Lo que tengo - Lo que necesito)
    # Nota: Asumimos que si no está en objetivo, todo es sobrante, 
    # o si está en objetivo y tengo más, el resto es sobrante.
    for k, v in recursos.items():
        necesario = objetivo.get(k, 0)
        diff = v - necesario
        if diff > 0:
            sobrantes[k] = diff
            
    return faltantes, sobrantes

# --------------
# -- ACCIONES --
# --------------

def enviar_carta(dest, asunto, cuerpo):
    url = f"{URL_BASE}/carta"
    data = {
        "remi": AGENT_NAME,
        "dest": dest,
        "asunto": asunto,
        "cuerpo": cuerpo
        }
    r = requests.post(url, json=data)

    if r.status_code == 200:
        print(f"Carta enviada correctamente a {dest}")
        print(cuerpo)
    else:
        print(f"Error al enviar carta a {dest}. Status: {r.status_code}, Respuesta: {r.text}")

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

def borrar_carta(id_carta: str):
    """Elimina una carta del buzón por su ID."""
    url = f"{URL_BASE}/mail/{id_carta}"
    try:
        r = requests.delete(url)
        if r.status_code == 200:
            print(f"🗑️ Carta {id_carta} borrada correctamente.")
        else:
            print(f"⚠️ No se pudo borrar la carta {id_carta}. Status: {r.status_code}")
    except Exception as e:
        print(f"❌ Error de conexión al borrar carta: {e}")

def enviar_paquete(destinatario, objeto, cantidad):
    url = f"{URL_BASE}/paquete/{destinatario}"
    data = {objeto: int(cantidad)}
    r = requests.post(url, data)
    if r.status_code == 200:
        print(f"📦 Paquete enviado a {destinatario}: {cantidad} de {objeto}")
    else:
        print(f"❌ Error enviando paquete: {r.text}")

def ejecutar_accion(accion_json):
    if not accion_json: return

    tipo = accion_json.get("accion")
    
    if tipo == "enviar_carta":
        # Construcción del cuerpo de la carta
        dest = accion_json.get("destinatario")
        req = accion_json.get("recurso_solicitado", "nada")
        req_cant = accion_json.get("cantidad_recurso_solicitado", 0)
        ofr = accion_json.get("recurso_ofrecido", "nada")
        ofr_cant = accion_json.get("cantidad_recurso_ofrecido", 0)
        
        cuerpo = (f"Hola {dest}, soy {AGENT_NAME}. "
                  f"Necesito {req_cant} de {req}. "
                  f"A cambio te ofrezco {ofr_cant} de {ofr}. "
                  f"Si te interesa, envíame el paquete.")
        
        enviar_carta(dest, "Propuesta de intercambio", cuerpo)

    elif tipo == "enviar_paquete":
        enviar_paquete(
            accion_json.get("destinatario"),
            accion_json.get("objeto_enviar"),
            accion_json.get("cantidad_enviar")
        )
    
    elif tipo == "esperar":
        print("⏳ El agente decide esperar...")

# --------------
# -- OLLAMA ---
# --------------

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
            options={'temperature': 0.1} # Temperatura baja para ser más preciso con JSON
        )
        content = response['message']['content']
        print(content)
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

# --------------
# -- MAIN ---
# --------------

def main():
    print("Iniciando agente IA...")

    # 1. ANÁLISIS INICIAL
    usuarios = get_gente()
    print(usuarios)

    if AGENT_NAME not in usuarios:
        register_agent(AGENT_NAME)
    else:
        print("Alias ya registrado.")

    info = get_info()
    if not info: return
    
    faltantes, sobrantes = calcular_estado(info)
    print(f"📊 Estado: Faltan {faltantes} | Sobran {sobrantes}")

    
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

        Recursos faltantes:
        {faltantes}

        Recursos sobrantes:
        {sobrantes}

        Usuarios:
        {usuarios}

        REGLAS DE RESPUESTA:
        - Responde SIEMPRE en JSON válido
        - No expliques nada fuera del JSON
        - Usa una de estas acciones:

        ACCIONES POSIBLES:
        1) enviar_carta
        2) enviar_paquete
        3) esperar

        FORMATO DE RESPUESTA OBLIGATORIO (JSON puro):
        
        "accion": "enviar_carta" | "enviar_paquete" | "esperar",
        "destinatario": "nombre_usuario",
        "recurso_solicitado": "nombre",
        "cantidad_recurso_solicitado": 0,
        "recurso_ofrecido": "nombre",
        "cantidad_recurso_ofrecido": 0
        
        """

    # 2. FASE DE DIFUSIÓN (Enviar cartas a todos)
    # print("\n--- 📨 FASE 1: Enviando propuestas a todos ---")

    # # Convertimos los dicts a listas para poder usar índices
    # lista_faltantes = list(faltantes.keys())
    # lista_sobrantes = list(sobrantes.keys())

    # for i, usuario in enumerate(usuarios):
    #     if usuario == AGENT_NAME: continue

    #     # 1. Alternamos el recurso necesitado usando el índice del bucle
    #     item_need = lista_faltantes[i % len(lista_faltantes)]
    #     cant_total_necesitada = faltantes[item_need]
        
    #     # Pedimos solo una parte o el total si es poco (ej. pedir de 5 en 5)
    #     cant_a_pedir = max(1, cant_total_necesitada // 2) 

    #     # 2. Alternamos el recurso ofrecido (si tenemos)
    #     if lista_sobrantes:
    #         item_offer = lista_sobrantes[i % len(lista_sobrantes)]
    #         total_disponible = sobrantes[item_offer]
            
    #         # ESTRATEGIA: No dar todo. Ofrecemos solo el 20% de lo que nos sobra
    #         # para tener margen de negociación con otros.
    #         cant_a_ofrecer = max(1, total_disponible // 5)
    #     else:
    #         item_offer = "nada"
    #         cant_a_ofrecer = 0
        
    #     prompt = f"""
    #     Genera una acción JSON para enviar una carta a '{usuario}'.
    #     CONTEXTO:
    #     - Solicitar: {cant_a_pedir} de {item_need}
    #     - Ofrecer a cambio: {cant_a_ofrecer} de {item_offer}
    #     """
        
    #     # Usamos el sys_prompt para que la respuesta sea JSON puro
    #     respuesta = ollama_generate(prompt, prompt_inicial) 
    #     ejecutar_accion(respuesta)
    #     time.sleep(1)

    # 3. FASE REACTIVA (Bucle infinito)
    print("\n--- 👁️ FASE 2: Esperando respuestas y paquetes ---")

    while True:

        info = get_info()
        buzon_dict = info.get("Buzon", [])

        # Convertimos el dict de dicts en una lista simple de cuerpos de mensaje
        mensajes_pendientes = []
        for carta_id, datos in buzon_dict.items():
            mensajes_pendientes.append({
                "de": datos["remi"],
                "asunto": datos["asunto"],
                "contenido": datos["cuerpo"]
            })

        # Chequear si ya ganamos (opcional)
        faltantes, _ = calcular_estado(info)
        if not faltantes:
            print("🏆 ¡OBJETIVO CUMPLIDO!")
            break

        # Si hay cartas, las procesamos
        if mensajes_pendientes:
            print(f"Tienes {len(mensajes_pendientes)} cartas nuevas.")
            # Leemos la última carta (simplificación)
            ultima_carta = mensajes_pendientes[-1] 
            print(ultima_carta)

            prompt = f"""
            CARTA RECIBIDA de {ultima_carta['de']}:
            "{ultima_carta['contenido']}"

            MIS RECURSOS: {info['Recursos']}
            MIS OBJETIVOS: {info['Objetivo']}

            INSTRUCCIÓN:
            Analiza si lo que ofrece {ultima_carta['de']} nos sirve para completar el objetivo.
            Por ejemplo, 'burrito sabanero' ofrece PIEDRA, y nosotros necesitamos 4 de PIEDRA.
            Si decides aceptar, responde con la acción 'enviar_paquete'.
            Si no aceptas, responde con la acción 'esperar'.
            """
            
            # Limpiamos el buzón (en una implementación real habría que borrar las cartas leídas o marcar como leídas)
            # Como la API proporcionada no tiene método de borrar explícito, asumimos que procesamos lo último.
            
            respuesta = ollama_generate(prompt, prompt_inicial)
            ejecutar_accion(respuesta)

            borrar_carta(carta_id)
            
        else:
            print("💤 Nada nuevo en el buzón. Esperando...")
        
        time.sleep(3)


if __name__ == "__main__":
    main()

# Registro con Butler

# While True

    # Miramos que gente hay endpoint gente

    # Intento conseguir recursos
