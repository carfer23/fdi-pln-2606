import ollama
from busqueda_clasica import busqueda_clasica
from busqueda_semantica import busqueda_semantica

from config import OLLAMA_MODEL

# MODELO_LLM = "llama3"
    
def ollama_chat(prompt: str, system_prompt: str = "") -> str | None:
    """Consulta a Ollama."""

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
    """Ejecuta búsqueda clásica y semántica y usa sus resultados como contexto para RAG."""

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
