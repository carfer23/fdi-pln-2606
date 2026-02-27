"""Módulo con métodos para realizar acciones."""

import requests
from pathlib import Path

from .config import AGENT_NAME, URL_BASE

P1_DIR = Path(__file__).resolve().parent


def register_agent(name: str):
    """Registra un alias."""

    url = f"{URL_BASE}/alias/{name}"
    r = requests.post(url, params={"agente": AGENT_NAME})

    if r.status_code == 200:
        print(f"Alias '{name}' registrado correctamente.")
    else:
        print(f"Alias '{name}' ya estaba registrado.")


def enviar_carta(dest, asunto, cuerpo):
    """Envía una carta."""

    url = f"{URL_BASE}/carta"
    data = {"remi": AGENT_NAME, "dest": dest, "asunto": asunto, "cuerpo": cuerpo}
    r = requests.post(url, json=data, params={"agente": AGENT_NAME})

    if r.status_code == 200:
        print(f"📨 Carta enviada correctamente a {dest}")
    else:
        print(
            f"❌ Error al enviar carta a {dest}. Status: {r.status_code}, Respuesta: {r.text}"
        )


def borrar_carta(id_carta: str):
    """Elimina una carta del buzón por su ID."""

    url = f"{URL_BASE}/mail/{id_carta}"
    try:
        r = requests.delete(url, params={"agente": AGENT_NAME})
        if r.status_code == 200:
            print(f"🗑️ Carta {id_carta} borrada correctamente.")
        else:
            print(f"⚠️ No se pudo borrar la carta {id_carta}. Status: {r.status_code}")
    except Exception as e:
        print(f"❌ Error de conexión al borrar carta: {e}")


def enviar_paquete(destinatario, objeto, cantidad):
    """Envía un paquete a otro usuario."""

    url = f"{URL_BASE}/paquete/{destinatario}"
    data = {objeto: int(cantidad)}

    try:
        r = requests.post(url, json=data, params={"agente": AGENT_NAME})

        if r.status_code == 200:
            print(f"📦 Paquete enviado a {destinatario}: {cantidad} de {objeto}")
        else:
            print(f"❌ Error al enviar el paquete: {r.text}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")


def cargar_carta(nombre_archivo, **kwargs):
    """Carga una plantilla de carta."""

    with open(f"{P1_DIR}/cartas/{nombre_archivo}.txt", "r", encoding="utf-8") as f:
        plantilla = f.read()
    return plantilla.format(**kwargs)


def ejecutar_accion(accion_json):
    """Ejecuta la acción elegida por el agente."""

    if not accion_json:
        return

    tipo = accion_json.get("accion")

    if tipo == "enviar_carta":
        print("📨 El agente envía una carta...")

        # Construcción del cuerpo de la carta
        dest = accion_json.get("destinatario")
        req = accion_json.get("recurso_solicitado", "nada")
        req_cant = accion_json.get("cantidad_recurso_solicitado", 0)
        ofr = accion_json.get("recurso_ofrecido", "nada")
        ofr_cant = accion_json.get("cantidad_recurso_ofrecido", 0)

        cuerpo = cargar_carta(
            "carta_difusion",
            dest=dest,
            alias=AGENT_NAME,
            req_cant=req_cant,
            req=req,
            ofr_cant=ofr_cant,
            ofr=ofr,
        )

        enviar_carta(dest, "Propuesta de intercambio", cuerpo)

    elif tipo == "enviar_paquete":
        print("📦 El agente acepta el trato y prepara el envío...")

        destinatario = accion_json.get("destinatario")
        recurso_enviar = accion_json.get("recurso_solicitado")
        cantidad_enviar = accion_json.get("cantidad_recurso_solicitado")
        recurso_esperado = accion_json.get("recurso_ofrecido")
        cantidad_esperada = accion_json.get("cantidad_recurso_ofrecido")

        if recurso_enviar and cantidad_enviar:
            enviar_paquete(destinatario, recurso_enviar, cantidad_enviar)

            # Enviar carta de confirmación
            cuerpo = cargar_carta(
                "carta_confirmacion",
                dest=destinatario,
                alias=AGENT_NAME,
                env_cant=cantidad_enviar,
                env_item=recurso_enviar,
                esp_cant=cantidad_esperada,
                esp_item=recurso_esperado,
            )

            enviar_carta(destinatario, "Paquete enviado", cuerpo)
            print(f"Carta de confirmación de envío: {cuerpo}")

    elif tipo == "esperar":
        print("⏳ El agente decide esperar...")
