import math
import json
import urllib.request
from procesador import tokenizar_query
from busqueda_semantica import obtener_capitulos_similares

MODELO_LLM = "llama3"
OLLAMA_URL = "http://localhost:11434"


def _ollama_chat(model, messages):
    """Llama a la API de Ollama para chat."""
    data = json.dumps({"model": model, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _calcular_idf(lema, capitulos):
    """Calcula el IDF (Inverse Document Frequency) de un lema."""
    n = len(capitulos)
    df = sum(1 for cap in capitulos if lema in cap["lemas"])
    if df == 0:
        return 0.0
    return math.log(n / df)


def busqueda_rag(consulta, capitulos):
    """Combina búsqueda clásica y semántica para obtener contexto,
    y usa un LLM a través de Ollama para generar una respuesta."""

    titulos_vistos = set()
    contextos = []

    # Capítulos de la búsqueda semántica
    caps_semantica = obtener_capitulos_similares(consulta, capitulos, top_k=3)
    for cap in caps_semantica:
        if cap["titulo"] not in titulos_vistos:
            titulos_vistos.add(cap["titulo"])
            contextos.append(f"[{cap['titulo']}]\n{cap['texto'][:1000]}")

    # Capítulos de la búsqueda clásica (los que no estén ya)
    palabras_busqueda = tokenizar_query(consulta)
    idfs = {lema: _calcular_idf(lema, capitulos) for lema in palabras_busqueda}
    caps_clasica = []
    for cap in capitulos:
        lemas_encontrados = palabras_busqueda & cap["lemas"]
        if lemas_encontrados:
            score = sum(cap["frecuencias"].get(l, 0) * idfs[l] for l in lemas_encontrados)
            caps_clasica.append((score, cap))
    caps_clasica.sort(key=lambda x: x[0], reverse=True)

    for _, cap in caps_clasica[:3]:
        if cap["titulo"] not in titulos_vistos:
            titulos_vistos.add(cap["titulo"])
            contextos.append(f"[{cap['titulo']}]\n{cap['texto'][:1000]}")

    contexto_completo = "\n\n---\n\n".join(contextos)

    prompt = f"""Eres un experto en El Quijote de Cervantes. Responde a la pregunta del usuario
basándote ÚNICAMENTE en los pasajes proporcionados. Si la respuesta no se encuentra en los pasajes,
indícalo. Responde en español. Al final de tu respuesta, incluye una sección "Fuentes:" listando
los títulos de los capítulos que has utilizado para responder.

PASAJES:
{contexto_completo}

PREGUNTA: {consulta}

RESPUESTA:"""

    respuesta = _ollama_chat(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
    )

    texto_respuesta = respuesta["message"]["content"]

    # Añadir referencias si el LLM no las incluyó
    if "fuente" not in texto_respuesta.lower():
        refs = "\n\nFuentes:\n" + "\n".join(f"  - {t}" for t in titulos_vistos)
        texto_respuesta += refs

    return texto_respuesta
