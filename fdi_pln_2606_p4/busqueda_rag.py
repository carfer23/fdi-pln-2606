import json
import urllib.request
import re
import unicodedata
from utils import nlp, similitud_coseno

MODELO_LLM = "llama3"
OLLAMA_URL = "http://localhost:11434"

TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
    "para", "con", "no", "una", "su", "al", "lo", "como", "mas", "pero", "sus", "le",
    "ya", "o", "este", "si", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre",
    "tambien", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante",
    "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
}

def _normalizar_token(token):
    token = token.lower()
    token = unicodedata.normalize("NFD", token)
    token = "".join(ch for ch in token if unicodedata.category(ch) != "Mn")
    return token

def _tokenizar(texto):
    tokens = []
    for token in TOKEN_RE.findall(texto):
        token_n = _normalizar_token(token)
        if len(token_n) > 2 and token_n not in STOPWORDS:
            tokens.append(token_n)
    return tokens

def obtener_capitulos_similares(consulta, capitulos, top_k=3):
    """Devuelve los capítulos más similares (como diccionarios) para uso interno (RAG)."""
    doc_consulta = nlp(consulta)
    emb_consulta = doc_consulta.vector

    similitudes = []
    for cap in capitulos:
        sim = similitud_coseno(emb_consulta, cap["embedding"])
        similitudes.append((sim, cap))

    similitudes.sort(key=lambda x: x[0], reverse=True)

    return [cap for _, cap in similitudes[:top_k]]

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

def busqueda_rag(query, capitulos):
    """Combina búsqueda clásica y semántica para obtener contexto,
    y usa un LLM a través de Ollama para generar una respuesta."""

    ranking_semantico = obtener_capitulos_similares(query, capitulos, top_k=3)

    tokens_q = set(_tokenizar(query))
    ranking_clasico = []
    for cap in capitulos:
        score = sum(cap["frecuencias"].get(token, 0) for token in tokens_q)
        if score > 0:
            ranking_clasico.append((score, cap))
    ranking_clasico.sort(key=lambda x: x[0], reverse=True)

    capitulos_contexto = []
    vistos = set()
    for cap in ranking_semantico:
        if cap["titulo"] not in vistos:
            vistos.add(cap["titulo"])
            capitulos_contexto.append(cap)
    for _, cap in ranking_clasico[:3]:
        if cap["titulo"] not in vistos:
            vistos.add(cap["titulo"])
            capitulos_contexto.append(cap)

    if not capitulos_contexto:
        return None

    contexto = "\n\n---\n\n".join(
        f"[{cap['titulo']}]\n{cap['texto'][:900]}" for cap in capitulos_contexto
    )
    prompt = (
        "Eres un experto en El Quijote. Responde en español usando solo los pasajes.\n"
        "Si no está en los pasajes, dilo claramente.\n"
        "Al final añade 'Fuentes:' con los títulos usados.\n\n"
        f"PASAJES:\n{contexto}\n\n"
        f"PREGUNTA: {query}\n\nRESPUESTA:"
    )

    respuesta = _ollama_chat(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
    )

    texto_respuesta = respuesta.get("message", {}).get("content")
    if not texto_respuesta:
        raise RuntimeError("Ollama no devolvió una respuesta válida para RAG.")

    return texto_respuesta
