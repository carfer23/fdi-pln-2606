"""Módulo con funciones de utilidad para el procesamiento de texto, generación de embeddings y cálculo de similitud."""

from bs4 import BeautifulSoup
import spacy
import math
import importlib.resources

from src.config import SPACY_MODEL

# Cargar modelo de spaCy
nlp = spacy.load(SPACY_MODEL)

# Configuración para división de texto en chunks
CHUNK_SIZE = 2000  # caracteres por chunk
OVERLAP = 200  # caracteres de solapamiento entre chunks


def crear_chunks(texto, chunk_size, overlap):
    """
    Divide un texto en fragmentos (chunks) de tamaño dictado, con solapamiento.

    :param texto: El texto completo a dividir.
    :param chunk_size: El tamaño máximo de cada chunk en caracteres.
    :param overlap: El número de caracteres que se solapan entre chunks consecutivos.
    :return: Lista de chunks de texto.
    """
    chunks = []
    start = 0
    texto_len = len(texto)
    while start < texto_len:
        end = start + chunk_size
        chunk = texto[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Avanzamos restando el solapamiento
    return chunks


def procesar_texto(texto):
    """Indexa un texto. Devuelve lemas y las posiciones (índices) en el texto original y frecuencias."""
    # Procesamos el texto con spaCy para obtener lemas y sus posiciones
    doc = nlp(texto)

    lemas = set()
    posiciones = {}  # Diccionario: lema -> lista de tuplas (inicio, fin)
    frecuencias = {}  # Diccionario: lema -> número de apariciones

    # Recorremos cada token para extraer lemas y sus posiciones
    for token in doc:
        lema = token.lemma_.lower()
        # Se ignoran stop words, puntuación o espacios vacíos
        if not token.is_stop and not token.is_punct and lema.strip():
            lemas.add(lema)
            if lema not in posiciones:
                posiciones[lema] = []
                frecuencias[lema] = 0
            # Guardamos dónde empieza y dónde acaba la palabra original
            posiciones[lema].append((token.idx, token.idx + len(token.text)))
            frecuencias[lema] += 1

    return lemas, posiciones, frecuencias


def separar_capitulos():
    """Lee el HTML del Quijote, separa los capítulos y devuelve una lista de diccionarios con
    id, título, texto completo, lemas, posiciones y frecuencias."""
    try:
        html = (
            importlib.resources.files("src")
            .joinpath("2000-h.htm")
            .read_text(encoding="utf-8")
        )
    except Exception:
        with open("2000-h.htm", "r", encoding="utf-8") as f:
            html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    capitulos = []

    for h3 in soup.find_all("h3"):
        a = h3.find("a")

        if a and a.get("name"):
            nombre = a["name"]

            if "_" in nombre:
                titulo = h3.get_text(strip=True)
                texto = []
                nodo = h3.find_next_sibling()

                while nodo and nodo.name != "h3":
                    if nodo.name == "p":
                        # Extraer texto del párrafo eliminando saltos de línea internos
                        parrafo_limpio = " ".join(
                            nodo.get_text(" ", strip=True).split()
                        )
                        if parrafo_limpio:
                            texto.append(parrafo_limpio)
                    nodo = nodo.find_next_sibling()

                texto_completo = "\n\n".join(texto)

                # Procesamos y extraemos posiciones y frecuencias de los lemas por chunk
                chunks = crear_chunks(texto_completo, CHUNK_SIZE, OVERLAP)

                for i, chunk in enumerate(chunks):
                    lemas, posiciones, frecuencias = procesar_texto(chunk)

                    capitulos.append(
                        {
                            "id": f"{nombre}_part{i + 1}",
                            "titulo": f"{titulo}",
                            "texto": chunk,
                            "lemas": lemas,
                            "posiciones": posiciones,
                            "frecuencias": frecuencias,
                        }
                    )

    return capitulos


def tokenizar_query(texto):
    """Solo para la consulta del usuario. Devuelve un set de lemas."""
    # Se procesa la consulta con spaCy
    doc = nlp(texto)

    return set(
        [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.lemma_.strip()
        ]
    )


def similitud_coseno(a, b):
    """Calcula la similitud coseno entre dos vectores."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def similitud_coseno_spacy(a, b):
    """Calcula la similitud coseno entre dos vectores con spaCy."""
    return a.similarity(b)
