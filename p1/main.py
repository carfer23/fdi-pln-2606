"""Módulo principal."""

import time

from info import get_info, get_gente
from acciones import register_agent, calcular_estado, ejecutar_accion, borrar_carta
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

    info = get_info()
    if not info: return
    
    faltantes, sobrantes = calcular_estado(info)
    print(f"📊 Estado: Faltan {faltantes} | Sobran {sobrantes}")
    
    prompt_inicial = cargar_prompt("prompt_inicial", 
                                   alias=AGENT_NAME, 
                                   faltantes=faltantes, 
                                   sobrantes=sobrantes,
                                   usuarios=usuarios)
    # -----------------------------------------------------------

    # ----- FASE DE DIFUSIÓN DE CARTAS --------------------------
    print("\n--- 📨 FASE 1: Enviando propuestas a todos ---")

    # Convertimos los dicts a listas para poder usar índices
    lista_faltantes = list(faltantes.keys())
    lista_sobrantes = list(sobrantes.keys())

    for i, usuario in enumerate(usuarios):
        if usuario == AGENT_NAME: continue

        print(f"Enviando carta a {usuario}...")

        # 1. Alternamos el recurso necesitado usando el índice del bucle
        item_need = lista_faltantes[i % len(lista_faltantes)]
        cant_total_necesitada = faltantes[item_need]
        
        # Pedimos solo una parte o el total si es poco (ej. pedir de 5 en 5)
        cant_a_pedir = max(1, cant_total_necesitada // 2) 

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
        
        respuesta = ollama_generate(prompt, prompt_inicial)
        ejecutar_accion(respuesta)
        time.sleep(1)
    # -----------------------------------------------------------

    # 3. FASE REACTIVA (Bucle infinito)
    print("\n--- 👁️ FASE 2: Esperando respuestas y paquetes ---")

    while True:

        info = get_info()
        buzon_dict = info.get("Buzon", [])

        # Convertimos el dict de dicts en una lista simple de cuerpos de mensaje
        mensajes_pendientes = []
        for carta_id, datos in buzon_dict.items():
            mensajes_pendientes.append({
                "de": datos["remi"],
                "asunto": datos["asunto"],
                "contenido": datos["cuerpo"]
            })

        # Chequear si ya ganamos (opcional)
        faltantes, _ = calcular_estado(info)
        if not faltantes:
            print("🏆 ¡OBJETIVO CUMPLIDO!")
            break

        # Si hay cartas, las procesamos
        if mensajes_pendientes:
            print(f"Tienes {len(mensajes_pendientes)} cartas nuevas.")
            # Leemos la última carta (simplificación)
            ultima_carta = mensajes_pendientes[-1] 
            print(ultima_carta)

            prompt = f"""
            CARTA RECIBIDA de {ultima_carta['de']}:
            "{ultima_carta['contenido']}"

            MIS RECURSOS: {info['Recursos']}
            MIS OBJETIVOS: {info['Objetivo']}

            INSTRUCCIÓN:
            Analiza si lo que ofrece {ultima_carta['de']} nos sirve para completar el objetivo.
            Por ejemplo, 'burrito sabanero' ofrece PIEDRA, y nosotros necesitamos 4 de PIEDRA.
            Si decides aceptar, responde con la acción 'enviar_paquete'.
            Si no aceptas, responde con la acción 'esperar'.
            """
            
            # Limpiamos el buzón (en una implementación real habría que borrar las cartas leídas o marcar como leídas)
            # Como la API proporcionada no tiene método de borrar explícito, asumimos que procesamos lo último.
            
            respuesta = ollama_generate(prompt, prompt_inicial)
            ejecutar_accion(respuesta)

            borrar_carta(carta_id)
            
        else:
            print("💤 Nada nuevo en el buzón. Esperando...")
        
        time.sleep(3)


if __name__ == "__main__":
    main()
