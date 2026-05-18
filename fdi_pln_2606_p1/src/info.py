"""Módulo para obtener el estado del agente desde el servidor butler."""

import requests
from typing import Dict

from config import URL_BASE, AGENT_NAME
from models import EstadoRecursos
from utils import console


def get_gente() -> list[str]:
    """Mira los alias registrados en el servidor y devuelve una lista con los nombres.

    :return: lista de alias de los agentes registrados en el servidor
    """
    url = f"{URL_BASE}/gente"

    try:
        r = requests.get(url, params={"agente": AGENT_NAME}, timeout=10)
        r.raise_for_status()
        gente = r.json()

        return [user.get("alias") if isinstance(user, dict) else user for user in gente]
    except requests.RequestException as e:
        console.print(f"[error]❌ Error de conexión al obtener usuarios: {e}[/error]")
        return []


def get_info() -> dict | None:
    """Obtiene toda la info del agente de una vez.

    :return: diccionario con la información del agente
    """
    try:
        r = requests.get(f"{URL_BASE}/info", params={"agente": AGENT_NAME}, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        console.print(f"[error]❌ Error conectando para obtener info: {e}[/error]")
        return None


def get_buzon() -> Dict:
    """Obtiene el contenido completo del buzón."""
    info = get_info()
    if not info:
        return {}
    return info.get("Buzon", {})


def calcular_estado() -> EstadoRecursos:
    """A partir de los recursos disponibles y los recursos objetivo,
    calcula los recursos faltantes y sobrantes. Devuelve dos diccionarios.

    :return: (faltantes, sobrantes)
    """
    info = get_info()

    if not info:
        return EstadoRecursos()

    recursos = info.get("Recursos", {})
    objetivo = info.get("Objetivo", {})

    faltantes = {
        k: v - recursos.get(k, 0)
        for k, v in objetivo.items()
        if v - recursos.get(k, 0) > 0
    }
    sobrantes = {
        k: v - objetivo.get(k, 0)
        for k, v in recursos.items()
        if v - objetivo.get(k, 0) > 0
    }

    return EstadoRecursos(faltantes=faltantes, sobrantes=sobrantes)
