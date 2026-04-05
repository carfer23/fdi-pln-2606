"""
Este módulo implementa un motor de búsqueda clásica basado en similitud léxica,
utilizando el modelo TF-IDF (Term Frequency - Inverse Document Frequency).

El algoritmo de búsqueda sigue este proceso:
1. Tokeniza y lematiza la consulta del usuario para unificar los términos.
2. Calcula el valor IDF (Inverse Document Frequency) de cada término, dando mayor peso a las palabras más raras en la colección.
3. Para cada capítulo que contiene alguno de los términos, calcula su puntuación sumando el TF-IDF (TF * IDF) de cada término coincidente.
4. Extrae un fragmento de contexto centrado en la primera aparición del término principal y resalta las palabras encontradas.
5. Ordena todos los resultados por su puntuación final (score) de mayor a menor relevancia.
"""

from utils import tokenizar_query
import math

def busqueda_clasica(query, capitulos):
    """
    Busca la palabra (o palabras) en los capítulos. Devuelve una lista de tuplas (score, título, fragmento).
    
    :param query: La consulta ingresada por el usuario.
    :param capitulos: La lista de capítulos procesados con lemas, posiciones y frecuencias.
    :return: Lista de tuplas (score, título, fragmento) ordenada por relevancia.
    """
    resultados = []
    palabras_busqueda = tokenizar_query(query)

    if not palabras_busqueda:
        return []
    
    # Calcular IDF para cada término de búsqueda
    N = len(capitulos)
    idf = {}
    for lema in palabras_busqueda:
        df = sum(1 for cap in capitulos if lema in cap.get("lemas", set()))
        idf[lema] = math.log(N / df) if df > 0 else 0

    for cap in capitulos:
        # Intersección de conjuntos. 
        lemas_encontrados = palabras_busqueda & cap.get("lemas", set())
        
        if lemas_encontrados:
            # Calcular TF-IDF para el capítulo
            score = 0
            # Si utils.py provee frecuencias usamos eso, sino contamos posiciones
            frecuencias = cap.get("frecuencias", {})
            for lema in lemas_encontrados:
                tf = frecuencias.get(lema)
                if tf is None and "posiciones" in cap and lema in cap["posiciones"]:
                    tf = len(cap["posiciones"][lema])
                elif tf is None:
                    tf = 1
                score += tf * idf[lema]
            
            lema_pivote = list(lemas_encontrados)[0]
            idx_inicio, idx_fin = cap["posiciones"][lema_pivote][0]
            
            # Se muestra un fragmento de texto alrededor de la primera aparición del lema encontrado
            ventana = 150
            inicio_contexto = max(0, idx_inicio - ventana)
            fin_contexto = min(len(cap["texto"]), idx_fin + ventana)
            
            fragmento_resaltado = cap["texto"][inicio_contexto:fin_contexto]
            
            # Recopilar las palabras exactas que aparecen en este fragmento (sin duplicados) para resaltarlas
            palabras_a_resaltar = set()
            for lema in lemas_encontrados:
                for pos_ini, pos_fin in cap["posiciones"][lema]:
                    if pos_ini >= inicio_contexto and pos_fin <= fin_contexto:
                        palabra_original = cap["texto"][pos_ini:pos_fin]
                        palabras_a_resaltar.add(palabra_original)

            # Ordenar las palabras de mayor a menor longitud
            # Evita que al reemplazar "viento" rompamos "vientos"
            palabras_a_resaltar = sorted(list(palabras_a_resaltar), key=len, reverse=True)

            # Resaltar cada palabra encontrada en el fragmento en color verde
            for palabra_original in palabras_a_resaltar:
                fragmento_resaltado = fragmento_resaltado.replace(
                    palabra_original, 
                    f"[b green]{palabra_original}[/b green]"
                )

            # Agregar "..." si el fragmento no muestra el inicio o el final del capítulo
            prefijo = "..." if inicio_contexto > 0 else ""
            sufijo = "..." if fin_contexto < len(cap["texto"]) else ""
            
            # Construir el texto final con los fragmentos resaltados y los "..."
            texto_final = f"{prefijo}{fragmento_resaltado}{sufijo}"
            
            # Agregar el resultado a la lista de resultados
            resultados.append((score, cap["titulo"], texto_final))
            
    # Ordenar por relevancia TF-IDF
    resultados.sort(key=lambda x: x[0], reverse=True)

    return resultados
