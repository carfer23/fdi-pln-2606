"""
Este módulo implementa un motor de búsqueda semántica basado en embeddings vectoriales.
Utiliza un modelo de lenguaje (a través de spaCy) para capturar el significado contextual del texto,
permitiendo encontrar resultados relevantes incluso si no coinciden las palabras exactas.

El algoritmo de búsqueda sigue este proceso:
1. Precomputa un embedding (representación vectorial) para cada capítulo a partir de sus tokens.
2. Convierte la consulta del usuario en un embedding dentro del mismo espacio vectorial.
3. Calcula la similitud entre el embedding de la consulta y los de todos los capítulos.
4. Selecciona los 'top_k' capítulos con mayor índice de similitud general.
5. Para cada capítulo seleccionado, extrae sus párrafos y calcula la similitud de cada uno con la consulta para encontrar el fragmento más pertinente.
6. Devuelve los resultados ordenados, incluyendo el porcentaje de similitud, el título del capítulo y el párrafo más representativo.
"""

import re
from src.utils import nlp, similitud_coseno

MAX_OUTPUT = 500
MAX_TEXT = 300


def generar_embeddings(capitulos):
    """
    Genera embeddings para cada capítulo usando spaCy.

    :param capitulos: Lista de diccionarios con información de cada capítulo.
    :return: Lista de embeddings correspondientes a cada capítulo.
    """
    embeddings = []
    for cap in capitulos:
        texto = cap["texto"]
        doc = nlp(texto)
        cap["embedding"] = doc.vector  # Embedding del capítulo (promedio de sus tokens)
        embeddings.append(doc.vector)

    return embeddings


def busqueda_semantica(consulta, capitulos, top_k=5):
    """
    Busca los capítulos más similares a la consulta. Por cada capítulo, también busca el párrafo más relevante.
    Devuelve una lista de tuplas (título, fragmento).

    :param consulta: Texto de la consulta del usuario.
    :param capitulos: Lista de diccionarios con información de cada capítulo, incluyendo su embedding.
    :param top_k: Número de capítulos más similares a devolver.
    :return: Lista de tuplas (título, fragmento) ordenada por relevancia.
    """
    # Procesar la consulta con spaCy para obtener su embedding
    doc_consulta = nlp(consulta)  # Se tokeniza internamente
    emb_consulta = doc_consulta.vector

    similitudes = []
    # Calcular similitud entre la consulta y cada capítulo
    for cap in capitulos:
        sim = similitud_coseno(emb_consulta, cap["embedding"])
        similitudes.append((sim, cap))

    # Ordenar por similitud descendente
    similitudes.sort(key=lambda x: x[0], reverse=True)

    resultados = []
    for sim, cap in similitudes[:top_k]:
        # Extraer párrafos válidos (ignorando líneas vacías o excesivamente cortas)
        parrafos = [
            p.strip() for p in re.split(r"\n+", cap["texto"]) if len(p.strip()) > 30
        ]
        if not parrafos:
            parrafos = [
                cap["texto"][:MAX_TEXT]
            ]  # Si no hay párrafos, se usa el inicio del texto

        mejor_sim_p = -1
        mejor_parrafo = parrafos[0]

        # Encontrar el párrafo más relevante para la consulta
        for p in parrafos:
            emb_p = nlp(p).vector
            sim_p = similitud_coseno(emb_consulta, emb_p)
            if sim_p > mejor_sim_p:
                mejor_sim_p = sim_p
                mejor_parrafo = p

        # Devolver el párrafo completo
        fragmento = f"... {mejor_parrafo} ..."

        porcentaje = sim * 100
        resultados.append((porcentaje, cap["titulo"], mejor_parrafo))

    return resultados
