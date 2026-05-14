"""Módulo con métodos para realizar acciones sobre el servidor butler."""

import requests
from pathlib import Path
from utils import console
from models import EstadoRecursos

from config import AGENT_NAME, URL_BASE

P1_DIR = Path(__file__).resolve().parent


def register_agent(name: str) -> None:
    """Registra un alias en el servidor.
    
    :param name: Alias a registrar
    """
    url = f"{URL_BASE}/alias/{name}"
    try:
        r = requests.post(url, params={"agente": AGENT_NAME}, timeout=10)
        
        if r.status_code == 200:
            console.print(f"[success]✅ Alias '{name}' registrado.[/success]")
        elif r.status_code == 400 and "ya existe" in r.text.lower():
            console.print(f"[info]ℹ️ Alias '{name}' ya estaba registrado.[/info]")
        else:
            r.raise_for_status()
            
    except requests.RequestException as e:
        console.print(f"[error]❌ Error registrando alias '{name}': {e}[/error]")


def cargar_carta(nombre_archivo: str, **kwargs) -> str:
    """Carga y formatea una plantilla de carta."""
    path = P1_DIR / "cartas" / f"{nombre_archivo}.txt"
    try:
        return path.read_text(encoding="utf-8").format(**kwargs)
    except FileNotFoundError:
        console.print(f"[error]❌ Plantilla no encontrada: {path}[/error]")
        return ""
    except KeyError as exc:
        console.print(f"[error]❌ Variables faltantes en plantilla {path}: {exc}[/error]")
        return ""


def enviar_carta(dest: str, asunto: str, cuerpo: str) -> None:
    """Envía una carta a otro agente.
    
    :param dest: Alias del destinatario
    :param asunto: Asunto de la carta
    :param cuerpo: Cuerpo de la carta
    """
    url = f"{URL_BASE}/carta"
    data = {"remi": AGENT_NAME, "dest": dest, "asunto": asunto, "cuerpo": cuerpo}
    try:
        r = requests.post(url, json=data, params={"agente": AGENT_NAME}, timeout=10)
        r.raise_for_status()
        console.print(f"[success]📨 Carta enviada a {dest}: '{asunto}'[/success]")
    except requests.RequestException as e:
        console.print(f"[error]❌ Error de conexión al enviar carta a {dest}: {e}[/error]")


def borrar_carta(id_carta: str) -> None:
    """Elimina una carta del buzón por su ID.
    
    :param id_carta: ID de la carta a eliminar
    """
    url = f"{URL_BASE}/mail/{id_carta}"
    try:
        r = requests.delete(url, params={"agente": AGENT_NAME}, timeout=10)
        r.raise_for_status()
        console.print(f"[success]🗑️ Carta {id_carta} eliminada.[/success]")
    except requests.RequestException as e:
        console.print(f"[error]❌ Error de conexión al borrar carta: {e}[/error]")


def enviar_paquete(destinatario: str, objeto: str, cantidad: int) -> bool:
    """Envía un paquete de recursos a otro agente. Devuelve True si tuvo éxito.
    
    :param destinatario: Alias del agente destinatario
    :param objeto: Nombre del recurso a enviar
    :param cantidad: Cantidad del recurso a enviar
    :return: True si el paquete se envió correctamente, False en caso de error
    """
    url = f"{URL_BASE}/paquete/{destinatario}"
    try:
        r = requests.post(url, json={objeto: cantidad}, params={"agente": AGENT_NAME})

        if r.status_code == 200:
            console.print(f"[success]📦 Paquete enviado a {destinatario}: {cantidad}x {objeto}[/success]")
            return True
        console.print(f"[error]❌ Error enviando paquete: {r.text}[/error]")
        return False
    except Exception as e:
        console.print(f"[error]❌ Error de conexión al enviar paquete: {e}[/error]")
        return False


def ejecutar_accion(accion_json: dict, estado: EstadoRecursos, asunto_recibido: str | None = None) -> None:
    """Ejecuta la acción elegida por el agente.
    
    :param accion_json: Diccionario con la acción a ejecutar, siguiendo el formato definido en DecisionAgente
    :param estado: EstadoRecursos con faltantes y sobrantes
    :param asunto_recibido: Asunto de la carta que estamos procesando, si aplica
    """
    if not isinstance(accion_json, dict):
        console.print("[warning]⚠️ Acción inválida (no es dict). Ignorando.[/warning]")
        return

    tipo = accion_json.get("accion")

    if tipo == "enviar_carta":
        console.print("[info]📨 El agente envía carta...[/info]")

        # Construcción del cuerpo de la carta
        dest = accion_json.get("destinatario")
        req = accion_json.get("recurso_recibir", "nada")
        req_cant = accion_json.get("cantidad_recurso_recibir", 0)
        ofr = accion_json.get("recurso_enviar", "nada")
        ofr_cant = accion_json.get("cantidad_recurso_enviar", 0)

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
        console.print("[info]📦 El agente acepta el trato y prepara el envío...[/info]")

        # Construcción del cuerpo de la carta
        destinatario = accion_json.get("destinatario")
        recurso_enviar = accion_json.get("recurso_enviar")
        cantidad_enviar = accion_json.get("cantidad_recurso_enviar")
        recurso_esperado = accion_json.get("recurso_recibir")
        cantidad_esperada = accion_json.get("cantidad_recurso_recibir")

        if not (destinatario and recurso_enviar and cantidad_enviar):
            console.print("[warning]⚠️ Acción enviar_paquete incompleta, ignorando.[/warning]")
            return

        # Limpiamos el nombre del recurso extraído por si el LLM coló texto de la plantilla anterior o cantidades.
        # Por ej: "trigo ( Sale de tus SOBRANTES)", o "1 de trigo" -> "trigo"
        recurso_limpio = recurso_enviar.split("(")[0].strip()
        if " de " in recurso_limpio:
            recurso_limpio = recurso_limpio.split(" de ")[-1].strip()
        recurso_limpio = ''.join(c for c in recurso_limpio if not c.isdigit()).strip()
        
        disponible = estado.sobrantes.get(recurso_limpio, 0)
        if int(disponible) < int(cantidad_enviar):
            console.print(f"[warning]⚠️ LLM quería enviar {cantidad_enviar}x '{recurso_limpio}' pero solo hay {disponible} en sobrantes. Ignorando.[/warning]")
        
        else:
            enviado_ok = enviar_paquete(destinatario, recurso_limpio, int(cantidad_enviar))
            if enviado_ok:
                asunto_lower = asunto_recibido.lower() if asunto_recibido else ""
                es_confirmacion = any(palabra in asunto_lower for palabra in ["confirm", "enviad", "camino"])
                
                if not es_confirmacion:
                    cuerpo = cargar_carta(
                        "carta_confirmacion",
                        dest=destinatario,
                        alias=AGENT_NAME,
                        env_cant=cantidad_enviar,
                        env_item=recurso_enviar,
                        esp_cant=cantidad_esperada,
                        esp_item=recurso_esperado
                    )

                    enviar_carta(destinatario, "Paquete enviado", cuerpo)
                    console.print(f"[success]Carta de confirmación de paquete enviada: {cuerpo}[/success]")
                else:
                    console.print(f"[info]Se omitió enviar carta de confirmación para evitar bucle (Asunto detectado: '{asunto_recibido}').[/info]")

    elif tipo == "esperar":
        console.print("[info]⏳ El agente decide no actuar.[/info]")

    else:
        console.print(f"[error]⚠️ Acción desconocida: '{tipo}'[/error]")
