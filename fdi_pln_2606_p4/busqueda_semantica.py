import re
from utils import nlp, similitud_coseno, tokenizar_query

MAX_CARACTERES = 6000
MAX_OUTPUT = 500
MAX_TEXT = 300

def generar_embeddings(capitulos):
    """Genera embeddings para cada capítulo usando spaCy.
    Modifica los diccionarios in-place añadiendo la clave 'embedding'."""
    embeddings = []
    for cap in capitulos:
        texto = cap["texto"][:MAX_CARACTERES]
        doc = nlp(texto)
        cap["embedding"] = doc.vector
        embeddings.append(doc.vector)
    
    return embeddings

def busqueda_semantica(consulta, capitulos, top_k=5):
    """Busca los capítulos más similares a la consulta. Por cada capítulo, también busca el párrafo más relevante.
    Devuelve una lista de tuplas (título, fragmento)."""
    tokens_query = tokenizar_query(consulta)
    consulta_filtrada = " ".join(tokens_query)
    
    doc_consulta = nlp(consulta_filtrada)
    emb_consulta = doc_consulta.vector

    similitudes = []
    for cap in capitulos:
        sim = similitud_coseno(emb_consulta, cap["embedding"])
        similitudes.append((sim, cap))

    similitudes.sort(key=lambda x: x[0], reverse=True)

    resultados = []
    for sim, cap in similitudes[:top_k]:
        # Dividir el capítulo en párrafos para extraer el más relevante
        parrafos = [p.strip() for p in re.split(r'\n+', cap["texto"])]
        if not parrafos:
            parrafos = [cap["texto"][:MAX_TEXT]] # Si no hay párrafos, se usa el inicio del texto
            
        mejor_sim_p = -1
        mejor_parrafo = parrafos[0]
        
        # Calcular el embedding de cada párrafo para buscar la misma similitud pero más granulada
        for p in parrafos:
            emb_p = nlp(p).vector
            sim_p = similitud_coseno(emb_consulta, emb_p)
            if sim_p > mejor_sim_p:
                mejor_sim_p = sim_p
                mejor_parrafo = p
                
        # Limitar la longitud del párrafo si es demasiado largo
        if len(mejor_parrafo) > MAX_OUTPUT:
            mejor_parrafo = mejor_parrafo[:MAX_OUTPUT] + "..."

        fragmento = f"... {mejor_parrafo} ..."

        porcentaje = sim * 100
        resultados.append((porcentaje, cap["titulo"], mejor_parrafo))

    return resultados
