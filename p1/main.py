"""Módulo principal."""

import time
import random

from info import get_gente, calcular_estado, get_buzon
from acciones import register_agent, ejecutar_accion, borrar_carta
from consulta_ollama import ollama_generate, cargar_prompt

from config import AGENT_NAME


def fase_difusion(faltantes, sobrantes, usuarios, prompt_inicial):
    """
    Fase 1: El agente envía cartas a otros usuarios para proponer intercambios.
    """
    print("\n--- 📨 FASE 1: Enviando cartas de difusión ---")

    # Convertimos los dicts a listas para poder usar índices
    lista_faltantes = list(faltantes.keys())
    lista_sobrantes = list(sobrantes.keys())

    # Si no nos falta nada, no iniciamos esta fase
    if not lista_faltantes:
        print("✅ No faltan recursos. Saltando fase de difusión.")
        return

    if len(usuarios) == 0:
        print("No hay otros usuarios activos para enviar cartas.")
        return

    for i, usuario in enumerate(usuarios):
        if usuario == AGENT_NAME:
            continue

        print(f"Preparando carta para {usuario}...")

        # 1. Seleccionamos aleatoriamente un recurso necesitado
        item_need = random.choice(lista_faltantes)
        cant_total_necesitada = faltantes[item_need]

        # Pedimos solo una parte
        # cant_a_pedir = max(1, cant_total_necesitada // 2)
        cant_a_pedir = 1

        # 2. Alternamos el recurso ofrecido (si tenemos sobrantes)
        if lista_sobrantes:
            item_offer = random.choice(lista_sobrantes)
            total_disponible = sobrantes[item_offer]

            # Estrategia: Ofrecer solo una parte para tener margen de negociación
            # cant_a_ofrecer = max(1, total_disponible // 5)
            cant_a_ofrecer = 1

            prompt = cargar_prompt(
                "prompt_difusion",
                usuario=usuario,
                cant_a_pedir=cant_a_pedir,
                item_need=item_need,
                cant_a_ofrecer=cant_a_ofrecer,
                item_offer=item_offer,
            )

            respuesta = ollama_generate(prompt, prompt_inicial)
            ejecutar_accion(respuesta)
            time.sleep(1)


def fase_reactiva(prompt_inicial, usuarios):
    """
    Fase 2: El agente reacciona a las cartas que llegan a su buzón.
    Si pasa un tiempo determinado sin recibir nada, vuelve a enviar cartas.
    """
    print("\n--- 👁️ FASE 2: Esperando respuestas y paquetes ---")

    TIMEOUT_DIFUSION = 15

    tiempo_ultima_accion = time.time()

    # Bucle infinito para mantener al agente activo
    while True:
        # Chequear si se ha cumplido el objetivo
        faltantes, sobrantes = calcular_estado()
        if not faltantes:
            print("🏆 ¡OBJETIVO CUMPLIDO! El agente ha conseguido todos los recursos.")
            break
        print(f"📊 Estado: Faltan {faltantes} | Sobran {sobrantes}")

        # Leer buzón
        buzon = get_buzon()

        if not buzon:
            print("💤 Buzón vacío. Esperando...")

            tiempo_actual = time.time()
            if (tiempo_actual - tiempo_ultima_accion) >= TIMEOUT_DIFUSION:
                print(
                    f"\n🔄 Han pasado {TIMEOUT_DIFUSION} segundos. Reenviando cartas de difusión..."
                )
                usuarios_actualizados = get_gente()
                fase_difusion(
                    faltantes, sobrantes, usuarios_actualizados, prompt_inicial
                )

                # Reiniciamos el temporizador después de enviar las cartas
                tiempo_ultima_accion = time.time()

            time.sleep(5)

        else:
            print(f"📫 Tienes {len(buzon)} mensajes en el buzón. Procesando...")

            # Extraer la primera carta
            if isinstance(buzon, dict):
                carta_id = next(iter(buzon))
                datos = buzon[carta_id]

                print(f"\nProcesando carta de {datos.get('remi')}...")

                prompt = cargar_prompt(
                    "prompt_procesar_carta",
                    usuario=datos.get("remi"),
                    contenido=datos.get("cuerpo"),
                    asunto=datos.get("asunto"),
                    faltantes=faltantes,
                    sobrantes=sobrantes,
                )

                respuesta = ollama_generate(prompt, prompt_inicial)
                ejecutar_accion(respuesta)

                # Eliminar la carta procesada
                borrar_carta(carta_id)

                tiempo_ultima_accion = time.time()
                time.sleep(5)
            else:
                print("⚠️ Formato de buzón inesperado.")


def main():
    """Función principal que orquesta el agente."""

    print("🚀 Iniciando agente IA...")

    # ----- CONFIGURACIÓN INICIAL -------------------------------
    usuarios = get_gente()

    if AGENT_NAME not in usuarios:
        register_agent(AGENT_NAME)
    else:
        print("Alias ya registrado.")

    faltantes, sobrantes = calcular_estado()
    print(f"📊 Estado: Faltan {faltantes} | Sobran {sobrantes}")

    prompt_inicial = cargar_prompt(
        "prompt_inicial",
        alias=AGENT_NAME,
        faltantes=faltantes,
        sobrantes=sobrantes,
        usuarios=usuarios,
    )

    fase_difusion(faltantes, sobrantes, usuarios, prompt_inicial)
    fase_reactiva(prompt_inicial, usuarios)


if __name__ == "__main__":
    main()
