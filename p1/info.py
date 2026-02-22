"""Módulo para los métodos que obtienen información del endpoint /info."""

import requests
import json

from config import URL_BASE

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
    """Obtiene toda la info del agente de una vez."""
    try:
        r = requests.get(f"{URL_BASE}/info")

        if r.status_code == 200:
            return json.loads(r.text)
    except Exception as e:
        print(f"Error conectando: {e}")
    return None
    
def get_recursos():
    """Obtiene los recursos actuales."""
    url = f"{URL_BASE}/info"
    r = requests.get(url)

    if r.status_code == 200:
        r = json.loads(r.text)
        return r["Recursos"]

def get_objetivo():
    """Obtiene los recursos objetivo."""
    url = f"{URL_BASE}/info"
    r = requests.get(url)

    if r.status_code == 200:
        r = json.loads(r.text)
        return r["Objetivo"]

def get_buzon():
    """Obtiene el contenido del buzón."""
    url = f"{URL_BASE}/info"
    r = requests.get(url)

    if r.status_code == 200:
        r = json.loads(r.text)
        return r["Buzon"]