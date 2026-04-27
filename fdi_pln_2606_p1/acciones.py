"""Módulo con métodos para realizar acciones sobre el servidor butler."""

import requests
from pathlib import Path

if __package__:
    from .config import AGENT_NAME, URL_BASE
else:
    from config import AGENT_NAME, URL_BASE

P1_DIR = Path(__file__).resolve().parent


def register_agent(name: str) -> None:
    """Registra un alias en el servidor."""
    try:
        r = requests.post(
            f"{URL_BASE}/alias/{name}",
            params={"agente": AGENT_NAME},
            timeout=10,
        )
        if r.status_code == 200:
            print(f"✅ Alias '{name}' registrado.")
        elif r.status_code == 409:
            print(f"ℹ️ Alias '{name}' ya estaba registrado.")
        else:
            print(f"❌ No se pudo registrar alias '{name}': {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"❌ Error registrando alias '{name}': {e}")


def enviar_carta(dest: str, asunto: str, cuerpo: str) -> None:
    """Envía una carta a otro agente."""
    data = {"remi": AGENT_NAME, "dest": dest, "asunto": asunto, "cuerpo": cuerpo}
    try:
        r = requests.post(f"{URL_BASE}/carta", json=data, params={"agente": AGENT_NAME})
        if r.status_code == 200:
            print(f"📨 Carta enviada a {dest}: '{asunto}'")
        else:
            print(f"❌ Error enviando carta a {dest}: {r.status_code} {r.text}")
    except Exception as e:
        print(f"❌ Error de conexión al enviar carta: {e}")


def borrar_carta(id_carta: str) -> None:
    """Elimina una carta del buzón por su ID."""
    try:
        r = requests.delete(f"{URL_BASE}/mail/{id_carta}", params={"agente": AGENT_NAME})
        if r.status_code == 200:
            print(f"🗑️ Carta {id_carta} eliminada.")
        else:
            print(f"⚠️ No se pudo eliminar la carta {id_carta}: {r.status_code}")
    except Exception as e:
        print(f"❌ Error de conexión al borrar carta: {e}")


def enviar_paquete(destinatario: str, objeto: str, cantidad: int) -> bool:
    """Envía un paquete de recursos a otro agente. Devuelve True si tuvo éxito."""
    try:
        r = requests.post(
            f"{URL_BASE}/paquete/{destinatario}",
            json={objeto: cantidad},
            params={"agente": AGENT_NAME},
        )
        if r.status_code == 200:
            print(f"📦 Paquete enviado a {destinatario}: {cantidad}x {objeto}")
            return True
        print(f"❌ Error enviando paquete: {r.text}")
        return False
    except Exception as e:
        print(f"❌ Error de conexión al enviar paquete: {e}")
        return False


def cargar_carta(nombre_archivo: str, **kwargs) -> str:
    """Carga y formatea una plantilla de carta."""
    path = P1_DIR / "cartas" / f"{nombre_archivo}.txt"
    return path.read_text(encoding="utf-8").format(**kwargs)


def ejecutar_accion(accion_json: dict) -> None:
    """Despacha la acción elegida por el agente."""
    tipo = accion_json.get("accion")

    if tipo == "enviar_carta":
        dest = accion_json.get("destinatario")
        req = accion_json.get("recurso_solicitado", "nada")
        req_cant = accion_json.get("cantidad_recurso_solicitado", 0)
        ofr = accion_json.get("recurso_ofrecido", "nada")
        ofr_cant = accion_json.get("cantidad_recurso_ofrecido", 0)

        cuerpo = cargar_carta(
            "carta_difusion",
            dest=dest, alias=AGENT_NAME,
            req_cant=req_cant, req=req,
            ofr_cant=ofr_cant, ofr=ofr,
        )
        enviar_carta(dest, "Propuesta de intercambio", cuerpo)

    elif tipo == "enviar_paquete":
        destinatario = accion_json.get("destinatario")
        recurso_enviar = accion_json.get("recurso_enviar")
        cantidad_enviar = accion_json.get("cantidad_recurso_enviar")
        recurso_esperado = accion_json.get("recurso_recibir")
        cantidad_esperada = accion_json.get("cantidad_recurso_recibir")

        if not (destinatario and recurso_enviar and cantidad_enviar):
            print("⚠️ Acción enviar_paquete incompleta, ignorando.")
            return

        enviado = enviar_paquete(destinatario, recurso_enviar, int(cantidad_enviar))
        if enviado:
            cuerpo = cargar_carta(
                "carta_confirmacion",
                dest=destinatario, alias=AGENT_NAME,
                env_cant=cantidad_enviar, env_item=recurso_enviar,
                esp_cant=cantidad_esperada, esp_item=recurso_esperado,
            )
            enviar_carta(destinatario, "Paquete enviado", cuerpo)

    elif tipo == "esperar":
        print("⏳ El agente decide no actuar.")

    else:
        print(f"⚠️ Acción desconocida: '{tipo}'")
