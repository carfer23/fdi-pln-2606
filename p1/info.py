"""Módulo para los métodos que obtienen información del endpoint /info."""

import requests
import json

from config import URL_BASE, AGENT_NAME


def get_gente():
    """Mira los alias registrados."""
    url = f"{URL_BASE}/gente"
    r = requests.get(url, params={"agente": AGENT_NAME})

    if r.status_code == 200:
        gente = json.loads(r.text)

        usuarios = []
        for user in gente:
            # Si el servidor devuelve un diccionario
            if isinstance(user, dict):
                usuarios.append(user.get("alias"))

            else:
                usuarios.append(user)

        return usuarios
    return []


def get_info():
    """Obtiene toda la info del agente de una vez."""
    try:
        r = requests.get(f"{URL_BASE}/info", params={"agente": AGENT_NAME})

        if r.status_code == 200:
            return json.loads(r.text)
    except Exception as e:
        print(f"Error conectando: {e}")
    return None


def calcular_estado():
    """Calcula qué sobra y qué falta."""
    info = get_info()

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


def get_buzon():
    """Obtiene el contenido del buzón."""
    info = get_info()
    return info.get("Buzon", [])
