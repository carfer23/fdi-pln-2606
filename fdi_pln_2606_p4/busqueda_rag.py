"""
Este módulo implementa un sistema de RAG (Retrieval-Augmented Generation).
Combina los resultados de los motores de búsqueda clásica y semántica para generar respuestas 
en lenguaje natural usando un modelo de lenguaje (LLM).

El proceso de búsqueda y generación es el siguiente:
1. Ejecuta la consulta del usuario en los sistemas de búsqueda clásica y semántica (embeddings).
2. Selecciona y filtra los mejores resultados de ambas búsquedas, asegurando variedad para no repetir capítulos.
3. Construye un contexto inyectando los fragmentos recuperados.
4. Elabora un prompt con directrices estrictas para que el LLM actúe como experto, 
   respondiendo únicamente basándose en el contexto dado.
5. Llama a un LLM local (Ollama) para generar la respuesta.
6. Garantiza que la respuesta contenga las fuentes estructuradas para dar trazabilidad a la información.
"""

import ollama
from busqueda_clasica import busqueda_clasica
from busqueda_semantica import busqueda_semantica

from config import OLLAMA_MODEL
    
def ollama_chat(prompt: str, system_prompt: str = "") -> str | None:
    """
    Consulta a un modelo de lenguaje de Ollama.
    
    :param prompt: El mensaje de usuario para el modelo.
    :param system_prompt: Instrucciones para el modelo (opcional).
    :return: La respuesta generada por el modelo, o None si hubo un error.
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        content = response["message"]["content"]

        return content

    except Exception:
        return None

def busqueda_rag(query, capitulos):
    """
    Ejecuta búsqueda clásica y semántica y usa sus resultados como contexto para RAG.
    
    :param query: La consulta del usuario.
    :param capitulos: Lista de diccionarios con información de cada capítulo, incluyendo embeddings
    :return: Respuesta generada por el modelo de lenguaje usando RAG, o None si no se pudo generar.
    """
    resultados_clasicos = busqueda_clasica(query, capitulos)
    resultados_semanticos = busqueda_semantica(query, capitulos, top_k=3)

    if not resultados_clasicos and not resultados_semanticos:
        return None

    capitulos_por_titulo = {cap["titulo"]: cap for cap in capitulos}

    contexto_bloques = []
    fuentes = []
    vistos = set()

    # Priorizamos los mejores resultados clásicos
    for score, titulo, fragmento in resultados_clasicos[:3]:
        if titulo in vistos:
            continue
        vistos.add(titulo)
        fuentes.append(titulo)
        contexto_bloques.append(
            f"[CLASICA | {titulo} | score={score:.4f}]\n{fragmento}"
        )

    # Añadimos semántica para complementar el contexto
    for porcentaje, titulo, fragmento in resultados_semanticos[:3]:
        if titulo in vistos:
            continue
        vistos.add(titulo)
        fuentes.append(titulo)

        # Si el fragmento viene vacío por cualquier motivo, usamos el texto del capítulo.
        if not fragmento and titulo in capitulos_por_titulo:
            fragmento = capitulos_por_titulo[titulo]["texto"]

        contexto_bloques.append(
            f"[SEMANTICA | {titulo} | similitud={porcentaje:.1f}%]\n{str(fragmento)}"
        )

    if not contexto_bloques:
        return None

    contexto = "\n\n---\n\n".join(contexto_bloques)
    prompt = (
        "Eres un experto en el texto de El Quijote de Miguel de Cervantes.\n" 
        "Responde a la consulta del usuario usando estrictamente el contexto recuperado.\n"
        "El contexto incluye fragmentos de resultados de búsqueda clásica y semántica.\n"
        "Si la respuesta no está respaldada por el contexto, indícalo claramente.\n"
        "Incluye citas directas o fragmentos textuales exactos del contexto para justificar tu respuesta, indicando de qué capítulo provienen.\n"
        "Al final añade 'Fuentes:' con los títulos de los capítulos usados del contexto.\n\n"
        f"CONTEXTO:\n{contexto}\n\nCONSULTA: {query}"
    )

    texto_respuesta = ollama_chat(prompt)
    if not texto_respuesta:
        raise RuntimeError("Ollama no devolvió una respuesta válida para RAG.")

    # Si el modelo no incluyó fuentes, las añadimos para garantizar trazabilidad.
    if "fuentes:" not in texto_respuesta.lower():
        texto_respuesta = f"{texto_respuesta}\n\nFuentes: {', '.join(fuentes)}"

    return texto_respuesta
