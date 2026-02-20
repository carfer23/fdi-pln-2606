"""Módulo con métodos para realizar acciones."""

import requests
import os

from config import AGENT_NAME

URL_BASE = os.getenv("FDI_PLN__BUTLER_ADDRESS")

def register_agent(name: str):
    """Registra un alias."""

    url = f"{URL_BASE}/alias/{name}"
    r = requests.post(url)

    if r.status_code == 200:
        print(f"Alias '{name}' registrado correctamente")
    else:
        print(f"Alias '{name}' ya estaba registrado")

def enviar_carta(dest, asunto, cuerpo):
    """Envía una carta."""

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
    """Genera la carta que el agente enviará a otro usuario solicitando intercambio de recursos."""

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
    """Envía un paquete."""

    url = f"{URL_BASE}/paquete/{destinatario}"
    data = {objeto: int(cantidad)}
    r = requests.post(url, data)
    if r.status_code == 200:
        print(f"📦 Paquete enviado a {destinatario}: {cantidad} de {objeto}")
    else:
        print(f"❌ Error enviando paquete: {r.text}")

def ejecutar_accion(accion_json):
    """Ejecuta la acción elegida por el agente."""

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

def calcular_estado(info):
    """Calcula qué sobra y qué falta."""
    
    recursos = info.get("Recursos", {})
    objetivo = info.get("Objetivo", {})
    
    faltantes = {}
    sobrantes = {}
    
    # Calcular faltantes (lo que necesito - lo que tengo)
    for k, v in objetivo.items():
        actual = recursos.get(k, 0)
        diff = v - actual
        if diff > 0:
            faltantes[k] = diff

    # Calcular sobrantes (lo que tengo - lo que necesito)
    # - Si no está en objetivo, todo es sobrante
    # - Si está en objetivo y tengo más, el resto es sobrante
    for k, v in recursos.items():
        necesario = objetivo.get(k, 0)
        diff = v - necesario
        if diff > 0:
            sobrantes[k] = diff
            
    return faltantes, sobrantes