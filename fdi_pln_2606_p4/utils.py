from bs4 import BeautifulSoup
import spacy
import math

# Cargar modelo de spaCy
nlp = spacy.load("es_core_news_sm")

# Configuración para división de texto en chunks
# CHUNK_SIZE = 2000  # caracteres por chunk
# OVERLAP = 200  # caracteres de solapamiento entre chunks

def procesar_texto_capitulo(texto):
    """Indexa un capítulo. Devuelve lemas y las posiciones (índices) en el texto original y frecuencias."""
    # Procesamos el texto con spaCy para obtener lemas y sus posiciones
    doc = nlp(texto)

    lemas = set()
    posiciones = {} # Diccionario: lema -> lista de tuplas (inicio, fin)
    frecuencias = {} # Diccionario: lema -> número de apariciones

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
                        texto.append(nodo.get_text(" ", strip=True))
                    nodo = nodo.find_next_sibling()

                texto_completo = " ".join(texto)
                
                # Procesamos y extraemos posiciones y frecuencias de los lemas
                lemas, posiciones, frecuencias = procesar_texto_capitulo(texto_completo)

                capitulos.append({
                    "id": nombre,
                    "titulo": titulo,
                    "texto": texto_completo,
                    "lemas": lemas,
                    "posiciones": posiciones,
                    "frecuencias": frecuencias,
                })

    return capitulos

def tokenizar_query(texto):
    """Solo para la consulta del usuario. Devuelve un set de lemas."""
    # Se procesa la consulta con spaCy
    doc = nlp(texto)

    return set([
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ])

def similitud_coseno(a, b):
    """Calcula la similitud coseno entre dos vectores."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
