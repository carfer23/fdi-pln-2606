import math
from procesador import tokenizar_query


def _calcular_idf(lema, capitulos):
    """Calcula el IDF (Inverse Document Frequency) de un lema."""
    n = len(capitulos)
    df = sum(1 for cap in capitulos if lema in cap["lemas"])
    if df == 0:
        return 0.0
    return math.log(n / df)


def busqueda_clasica(palabra, capitulos):
    """Busca la palabra (o palabras) en los capítulos.
    Ordena los resultados por relevancia TF-IDF.
    Devuelve una lista de tuplas (título, fragmento)."""
    palabras_busqueda = tokenizar_query(palabra)

    if not palabras_busqueda:
        return []

    # Pre-calcular IDF para cada término de la consulta
    idfs = {lema: _calcular_idf(lema, capitulos) for lema in palabras_busqueda}

    resultados_con_score = []

    for cap in capitulos:
        lemas_encontrados = palabras_busqueda & cap["lemas"]

        if not lemas_encontrados:
            continue

        # Calcular score TF-IDF: sum(tf * idf) para cada lema encontrado
        score = 0.0
        for lema in lemas_encontrados:
            tf = cap["frecuencias"].get(lema, 0)
            score += tf * idfs[lema]

        # Construir el fragmento resaltado centrado en el lema más relevante
        # (el de mayor TF-IDF individual)
        lema_pivote = max(lemas_encontrados, key=lambda l: cap["frecuencias"].get(l, 0) * idfs[l])
        idx_inicio, idx_fin = cap["posiciones"][lema_pivote][0]

        ventana = 150
        inicio_contexto = max(0, idx_inicio - ventana)
        fin_contexto = min(len(cap["texto"]), idx_fin + ventana)

        fragmento_resaltado = cap["texto"][inicio_contexto:fin_contexto]

        # Recopilar palabras exactas para resaltar
        palabras_a_resaltar = set()
        for lema in lemas_encontrados:
            for pos_ini, pos_fin in cap["posiciones"][lema]:
                if pos_ini >= inicio_contexto and pos_fin <= fin_contexto:
                    palabras_a_resaltar.add(cap["texto"][pos_ini:pos_fin])

        # Ordenar de mayor a menor longitud para evitar reemplazos parciales
        palabras_a_resaltar = sorted(list(palabras_a_resaltar), key=len, reverse=True)

        for palabra_original in palabras_a_resaltar:
            fragmento_resaltado = fragmento_resaltado.replace(
                palabra_original,
                f"[b green]{palabra_original}[/b green]"
            )

        prefijo = "..." if inicio_contexto > 0 else ""
        sufijo = "..." if fin_contexto < len(cap["texto"]) else ""

        texto_final = f"[dim](relevancia: {score:.2f})[/dim]\n{prefijo}{fragmento_resaltado}{sufijo}"

        resultados_con_score.append((score, cap["titulo"], texto_final))

    # Ordenar por score descendente
    resultados_con_score.sort(key=lambda x: x[0], reverse=True)

    return [(titulo, texto) for _, titulo, texto in resultados_con_score]
