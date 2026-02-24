"""Módulo principal."""

import time

from info import get_gente, calcular_estado, get_buzon
from acciones import register_agent, ejecutar_accion, borrar_carta
from consulta_ollama import ollama_generate, cargar_prompt

from config import AGENT_NAME

def main():
    print("Iniciando agente IA...")

    # ----- CONFIGURACIÓN INICIAL -------------------------------
    usuarios = get_gente()

    if AGENT_NAME not in usuarios:
        register_agent(AGENT_NAME)
    else:
        print("Alias ya registrado.")
    
    faltantes, sobrantes = calcular_estado()
    print(f"📊 Estado: Faltan {faltantes} | Sobran {sobrantes}")
    
    prompt_inicial = cargar_prompt("prompt_inicial", 
                                   alias=AGENT_NAME, 
                                   faltantes=faltantes, 
                                   sobrantes=sobrantes,
                                   usuarios=usuarios)

    # ----- FASE DE DIFUSIÓN DE CARTAS --------------------------
    print("\n--- 📨 FASE 1: Enviando cartas a todos ---")

    # Convertimos los dicts a listas para poder usar índices
    lista_faltantes = list(faltantes.keys())
    lista_sobrantes = list(sobrantes.keys())

    if len(usuarios) == 0:
        print("No hay otros usuarios activos.")

    for i, usuario in enumerate(usuarios):
        if usuario == AGENT_NAME: continue

        print(f"Enviando carta a {usuario}...")

        # 1. Alternamos el recurso necesitado usando el índice del bucle
        item_need = lista_faltantes[i % len(lista_faltantes)]
        cant_total_necesitada = faltantes[item_need]
        
        # Pedimos solo una parte o el total si es poco (ej. pedir de 5 en 5)
        #cant_a_pedir = max(1, cant_total_necesitada // 2) 
        cant_a_pedir = 1

        # 2. Alternamos el recurso ofrecido (si tenemos)
        if lista_sobrantes:
            item_offer = lista_sobrantes[i % len(lista_sobrantes)]
            total_disponible = sobrantes[item_offer]
            
            # ESTRATEGIA: No dar todo. Ofrecemos solo el 20% de lo que nos sobra
            # para tener margen de negociación con otros.
            cant_a_ofrecer = max(1, total_disponible // 5)
        else:
            item_offer = "nada"
            cant_a_ofrecer = 0
        
        prompt = cargar_prompt("prompt_difusion", 
                                usuario=usuario,
                                cant_a_pedir=cant_a_pedir, 
                                item_need=item_need, 
                                cant_a_ofrecer=cant_a_ofrecer,
                                item_offer=item_offer)
        
        print(f"PROMPT: {prompt}")
        
        respuesta = ollama_generate(prompt, prompt_inicial)
        print(respuesta)
        ejecutar_accion(respuesta)
        time.sleep(1)

    # ----- FASE REACTIVA ---------------------------------------
    print("\n--- 👁️ FASE 2: Esperando respuestas y paquetes ---")

    # Número de cartas pedientes de procesar
    num_cartas_pendientes = 0

    # Bucle infinito
    while True:

        # Chequear si se ha cumplido el objetivo
        faltantes, sobrantes = calcular_estado()
        if not faltantes:
            print("🏆 ¡OBJETIVO CUMPLIDO!")
            break
        print(f"📊 Estado: Faltan {faltantes} | Sobran {sobrantes}")
        
        # Leer buzón
        buzon_dict = get_buzon()

        mensajes_nuevos = []
        if len(buzon_dict) == 0:
            print("Buzón vacío.")
        else:
            for carta_id, datos in buzon_dict.items():
                mensajes_nuevos.append({
                    "de": datos["remi"],
                    "asunto": datos["asunto"],
                    "contenido": datos["cuerpo"]
                })

        # Si hay cartas nuevas o cartas pendientes -> las procesamos
        if mensajes_nuevos or num_cartas_pendientes > 0:
            # Número actual de cartas en el buzón
            num_cartas = len(mensajes_nuevos)
            print(f"Tienes {num_cartas-num_cartas_pendientes} cartas nuevas. TOTAL: {num_cartas} cartas en el buzón.")
            # Número actual de cartas pendientes de procesar
            num_cartas_pendientes = num_cartas

            # Leemos la primera carta (la más antigua)
            primera_carta = mensajes_nuevos[-1] 

            print(primera_carta)

            prompt = cargar_prompt("prompt_procesar_carta",
                                   usuario=primera_carta['de'],
                                   contenido=primera_carta['contenido'],
                                   asunto=primera_carta['asunto'],
                                   faltantes=faltantes,
                                   sobrantes=sobrantes)
            
            #print(f"PROMPT: {prompt}")
            
            respuesta = ollama_generate(prompt, prompt_inicial)
            ejecutar_accion(respuesta)

            # Eliminar carta procesada del buzón
            borrar_carta(carta_id)
            # Una carta pendiente menos
            num_cartas_pendientes -= 1
            
        else:
            print("💤 Nada nuevo en el buzón. Esperando...")
        
        time.sleep(3)


if __name__ == "__main__":
    main()
