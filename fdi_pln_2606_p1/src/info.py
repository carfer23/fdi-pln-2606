"""Módulo para obtener el estado del agente desde el servidor butler."""

import requests

if __package__:
    from .config import URL_BASE, AGENT_NAME
else:
    from config import URL_BASE, AGENT_NAME


def get_state() -> dict | None:
    """Obtiene el estado completo del agente en una única llamada al servidor.

    Devuelve un dict con 'faltantes', 'sobrantes' y 'buzon', o None si falla.
    """
    try:
        r = requests.get(f"{URL_BASE}/info", params={"agente": AGENT_NAME}, timeout=10)
        if r.status_code != 200:
            print(f"❌ /info devolvió {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        recursos = data.get("Recursos", {})
        objetivo = data.get("Objetivo", {})
        return {
            "faltantes": {
                k: v - recursos.get(k, 0)
                for k, v in objetivo.items()
                if v > recursos.get(k, 0)
            },
            "sobrantes": {
                k: v - objetivo.get(k, 0)
                for k, v in recursos.items()
                if v > objetivo.get(k, 0)
            },
            "buzon": data.get("Buzon", {}),
        }
    except Exception as e:
        print(f"❌ Error conectando al servidor: {e}")
        return None


def get_gente() -> list[str]:
    """Devuelve los alias registrados en el servidor."""
    try:
        r = requests.get(f"{URL_BASE}/gente", params={"agente": AGENT_NAME}, timeout=10)
        if r.status_code != 200:
            print(f"❌ /gente devolvió {r.status_code}: {r.text[:200]}")
            return []
        gente = r.json()
        return [u.get("alias") if isinstance(u, dict) else u for u in gente]
    except Exception as e:
        print(f"❌ Error obteniendo usuarios: {e}")
        return []
