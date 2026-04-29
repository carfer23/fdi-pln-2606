"""Módulo principal del agente de trueque."""

import time

if __package__:
    from .info import get_state, get_gente
    from .acciones import register_agent, enviar_carta, cargar_carta, borrar_carta, ejecutar_accion
    from .consulta_ollama import ollama_generate, cargar_prompt
    from .config import AGENT_NAME, URL_BASE, OLLAMA_MODEL
else:
    from info import get_state, get_gente
    from acciones import register_agent, enviar_carta, cargar_carta, borrar_carta, ejecutar_accion
    from consulta_ollama import ollama_generate, cargar_prompt
    from config import AGENT_NAME, URL_BASE, OLLAMA_MODEL

TIMEOUT_REDIFUSION = 15  # segundos de inactividad antes de reenviar difusión
SYSTEM_PROMPT_FILE = "prompt_inicial"


def _build_system_prompt() -> str:
    return cargar_prompt(SYSTEM_PROMPT_FILE, alias=AGENT_NAME)


def fase_difusion(faltantes: dict, sobrantes: dict, usuarios: list[str]) -> None:
    """Envía propuestas de intercambio a todos los agentes disponibles."""
    print("\n--- 📨 FASE 1: Enviando cartas de difusión ---")

    if not faltantes:
        print("No faltan recursos. Saltando difusión.")
        return
    if not sobrantes:
        print("Sin sobrantes para ofrecer. No se puede proponer intercambio.")
        return

    items_need = list(faltantes)
    items_offer = list(sobrantes)
    destinatarios = [u for u in usuarios if u != AGENT_NAME]

    if not destinatarios:
        print("No hay otros agentes disponibles.")
        return

    for usuario in destinatarios:
        for j, item_need in enumerate(items_need):
            item_offer = items_offer[j % len(items_offer)]
            cuerpo = cargar_carta(
                "carta_difusion",
                dest=usuario, alias=AGENT_NAME,
                req_cant=1, req=item_need,
                ofr_cant=1, ofr=item_offer,
            )
            enviar_carta(usuario, "Propuesta de intercambio", cuerpo)
            time.sleep(0.5)


def _procesar_buzon(buzon: dict[str, dict], system_prompt: str) -> None:
    """Procesa todas las cartas del buzón, refrescando el estado entre cada una."""
    print(f"📫 {len(buzon)} mensajes en el buzón. Procesando...")

    for carta_id, datos in buzon.items():
        # Refrescamos el estado para reflejar paquetes ya enviados en este ciclo
        state = get_state()
        if state is None or not state["faltantes"]:
            break

        remitente = datos.get("remi", "desconocido")
        asunto = datos.get("asunto", "")
        print(f"\n📩 Carta de {remitente} [{asunto}]")

        if remitente == "Sistema":
            print("ℹ️ Carta del Sistema, ignorando.")
            borrar_carta(carta_id)
            continue

        prompt = cargar_prompt(
            "prompt_procesar_carta",
            usuario=remitente,
            contenido=datos.get("cuerpo", ""),
            asunto=asunto,
            faltantes=state["faltantes"],
            sobrantes=state["sobrantes"],
        )

        accion = ollama_generate(prompt, system_prompt)

        if accion.get("accion") == "enviar_paquete":
            recurso = accion.get("recurso_enviar", "")
            cantidad = int(accion.get("cantidad_recurso_enviar") or 0)
            disponible = state["sobrantes"].get(recurso, 0)
            if disponible < cantidad:
                print(f"⚠️ LLM quería enviar {cantidad}x {recurso} pero solo hay {disponible} en sobrantes. Ignorando.")
                borrar_carta(carta_id)
                continue

        ejecutar_accion(accion)
        borrar_carta(carta_id)
        time.sleep(1)


def fase_reactiva(system_prompt: str, usuarios: list[str]) -> None:
    """Bucle reactivo: procesa el buzón y reenvía difusión si hay inactividad."""
    print("\n--- 👁️ FASE 2: Modo reactivo ---")
    tiempo_ultima_accion = time.time()

    while True:
        state = get_state()

        if state is None:
            print("⚠️ Sin conexión al servidor. Reintentando en 5s...")
            time.sleep(5)
            continue

        print(f"\n📊 Faltantes: {state['faltantes']} | Sobrantes: {state['sobrantes']}")

        if not state["faltantes"]:
            print("🏆 ¡OBJETIVO CUMPLIDO! Todos los recursos conseguidos.")
            break

        if state["buzon"]:
            _procesar_buzon(state["buzon"], system_prompt)
            tiempo_ultima_accion = time.time()
        else:
            print("💤 Buzón vacío.")
            if time.time() - tiempo_ultima_accion >= TIMEOUT_REDIFUSION:
                print(f"🔄 {TIMEOUT_REDIFUSION}s sin actividad. Reenviando difusión...")
                usuarios = get_gente()
                fase_difusion(state["faltantes"], state["sobrantes"], usuarios)
                tiempo_ultima_accion = time.time()

        time.sleep(5)


def main() -> None:
    """Orquesta el ciclo de vida del agente."""
    print("🚀 Iniciando agente IA...")
    if not URL_BASE:
        print("❌ Falta FDI_PLN__BUTLER_ADDRESS en .env")
        return
    if not AGENT_NAME:
        print("❌ Falta AGENT_NAME en .env")
        return
    if not OLLAMA_MODEL:
        print("❌ Falta OLLAMA_MODEL en .env")
        return

    usuarios = get_gente()
    if AGENT_NAME not in usuarios:
        register_agent(AGENT_NAME)

    state = get_state()
    if state is None:
        print("❌ No se pudo conectar al servidor. Abortando.")
        return

    print(f"📊 Estado inicial — Faltantes: {state['faltantes']} | Sobrantes: {state['sobrantes']}")

    system_prompt = _build_system_prompt()

    fase_difusion(state["faltantes"], state["sobrantes"], usuarios)
    fase_reactiva(system_prompt, usuarios)


if __name__ == "__main__":
    main()
