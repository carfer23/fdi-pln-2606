import math
import json
import urllib.request

MODELO_EMBEDDINGS = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"


def _ollama_embed(model, input_text):
    """Llama a la API de Ollama para generar embeddings."""
    data = json.dumps({"model": model, "input": input_text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _similitud_coseno(a, b):
    """Calcula la similitud coseno entre dos vectores."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


MAX_CARACTERES = 6000


def generar_embeddings(capitulos):
    """Genera embeddings para cada capítulo usando Ollama.
    Modifica los diccionarios in-place añadiendo la clave 'embedding'."""
    for cap in capitulos:
        texto = cap["texto"][:MAX_CARACTERES]
        respuesta = _ollama_embed(MODELO_EMBEDDINGS, texto)
        cap["embedding"] = respuesta["embeddings"][0]


def busqueda_embeddings(consulta, capitulos, top_k=5):
    """Busca los capítulos más similares a la consulta usando similitud coseno.
    Devuelve una lista de tuplas (título, fragmento) con los top_k resultados."""
    respuesta = _ollama_embed(MODELO_EMBEDDINGS, consulta)
    emb_consulta = respuesta["embeddings"][0]

    similitudes = []
    for cap in capitulos:
        sim = _similitud_coseno(emb_consulta, cap["embedding"])
        similitudes.append((sim, cap))

    similitudes.sort(key=lambda x: x[0], reverse=True)

    resultados = []
    for sim, cap in similitudes[:top_k]:
        fragmento = cap["texto"][:300]
        if len(cap["texto"]) > 300:
            fragmento += "..."

        porcentaje = sim * 100
        texto = f"[dim](similitud: {porcentaje:.1f}%)[/dim]\n{fragmento}"
        resultados.append((cap["titulo"], texto))

    return resultados


def obtener_capitulos_similares(consulta, capitulos, top_k=3):
    """Devuelve los capítulos más similares (como diccionarios) para uso interno (RAG)."""
    respuesta = _ollama_embed(MODELO_EMBEDDINGS, consulta)
    emb_consulta = respuesta["embeddings"][0]

    similitudes = []
    for cap in capitulos:
        sim = _similitud_coseno(emb_consulta, cap["embedding"])
        similitudes.append((sim, cap))

    similitudes.sort(key=lambda x: x[0], reverse=True)

    return [cap for _, cap in similitudes[:top_k]]
