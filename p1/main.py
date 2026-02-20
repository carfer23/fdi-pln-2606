"""Módulo principal."""

import time

from info import get_info, get_gente
from acciones import register_agent, calcular_estado, ejecutar_accion, borrar_carta
from consulta_ollama import ollama_generate

from config import AGENT_NAME

def main():
    print("Iniciando agente IA...")

    # ----- ANÁLISIS INICIAL -----
    usuarios = get_gente()
    print(usuarios)

    if AGENT_NAME not in usuarios:
        register_agent(AGENT_NAME)
    else:
        print("Alias ya registrado.")

    info = get_info()
    if not info: return
    
    faltantes, sobrantes = calcular_estado(info)
    print(f"📊 Estado: Faltan {faltantes} | Sobran {sobrantes}")
    # ----------------------------
    
    prompt_inicial = f"""
        Eres un agente que maneja recursos y los intercambia con otros usuarios.

        Tu objetivo es conseguir recolectar los recursos objetivo.
        Al inicio dispones de unos recursos determinados.

        Para conseguir los recursos objetivo:
        - debes enviar cartas para pedirlos
        - recibirás cartas de otros usuarios pidiendo recursos que puedes tener o no

        Para intercambiar recursos con otros usuarios debes enviar paquetes con esos recursos
        solo si con ello te acercas a tu objetivo.

        Tu usuario es: {AGENT_NAME}. No puedes enviarte cartas ni paquetes a ti mismo.

        Lo primero que tienes que hacer es enviar una carta a cada usuario solicitando los recursos que te faltan para cumplir
        el objetivo y los que puedes ofrecer. Por cada carta solo puede solicitar y ofrecer un recurso (la cantidad da igual).
        Para enviar una carta, en el JSON de respuesta indica la acción enviar_carta e indica el recurso a solicitar y su cantidad, 
        y el recurso a ofrecer y su cantidad.

        Recursos faltantes:
        {faltantes}

        Recursos sobrantes:
        {sobrantes}

        Usuarios:
        {usuarios}

        REGLAS DE RESPUESTA:
        - Responde SIEMPRE en JSON válido
        - No expliques nada fuera del JSON
        - Usa una de estas acciones:

        ACCIONES POSIBLES:
        1) enviar_carta
        2) enviar_paquete
        3) esperar

        FORMATO DE RESPUESTA OBLIGATORIO (JSON puro):
        
        "accion": "enviar_carta" | "enviar_paquete" | "esperar",
        "destinatario": "nombre_usuario",
        "recurso_solicitado": "nombre",
        "cantidad_recurso_solicitado": 0,
        "recurso_ofrecido": "nombre",
        "cantidad_recurso_ofrecido": 0
        
        """

    # 2. FASE DE DIFUSIÓN (Enviar cartas a todos)
    # print("\n--- 📨 FASE 1: Enviando propuestas a todos ---")

    # # Convertimos los dicts a listas para poder usar índices
    # lista_faltantes = list(faltantes.keys())
    # lista_sobrantes = list(sobrantes.keys())

    # for i, usuario in enumerate(usuarios):
    #     if usuario == AGENT_NAME: continue

    #     # 1. Alternamos el recurso necesitado usando el índice del bucle
    #     item_need = lista_faltantes[i % len(lista_faltantes)]
    #     cant_total_necesitada = faltantes[item_need]
        
    #     # Pedimos solo una parte o el total si es poco (ej. pedir de 5 en 5)
    #     cant_a_pedir = max(1, cant_total_necesitada // 2) 

    #     # 2. Alternamos el recurso ofrecido (si tenemos)
    #     if lista_sobrantes:
    #         item_offer = lista_sobrantes[i % len(lista_sobrantes)]
    #         total_disponible = sobrantes[item_offer]
            
    #         # ESTRATEGIA: No dar todo. Ofrecemos solo el 20% de lo que nos sobra
    #         # para tener margen de negociación con otros.
    #         cant_a_ofrecer = max(1, total_disponible // 5)
    #     else:
    #         item_offer = "nada"
    #         cant_a_ofrecer = 0
        
    #     prompt = f"""
    #     Genera una acción JSON para enviar una carta a '{usuario}'.
    #     CONTEXTO:
    #     - Solicitar: {cant_a_pedir} de {item_need}
    #     - Ofrecer a cambio: {cant_a_ofrecer} de {item_offer}
    #     """
        
    #     # Usamos el sys_prompt para que la respuesta sea JSON puro
    #     respuesta = ollama_generate(prompt, prompt_inicial) 
    #     ejecutar_accion(respuesta)
    #     time.sleep(1)

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
