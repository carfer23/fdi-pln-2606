from bs4 import BeautifulSoup
import spacy

# Cargar modelo de spaCy
nlp = spacy.load("es_core_news_sm")

CHUNK_SIZE = 2000  # caracteres por chunk
OVERLAP = 200  # caracteres de solapamiento entre chunks


def _dividir_en_chunks(texto, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Divide un texto en chunks con solapamiento, intentando cortar por frases (puntos)."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + chunk_size
        if fin >= len(texto):
            chunks.append((texto[inicio:], inicio))
            break
        # Intentar cortar en el último punto dentro del rango
        punto = texto.rfind(". ", inicio, fin)
        if punto > inicio:
            fin = punto + 2  # incluir el punto y espacio
        chunks.append((texto[inicio:fin], inicio))
        inicio = fin - overlap
    return chunks


def procesar_texto_capitulo(texto):
    """Indexa un capítulo. Devuelve lemas, posiciones y frecuencias."""
    lemas = set()
    posiciones = {}  # lema -> lista de tuplas (inicio, fin)
    frecuencias = {}  # lema -> número de apariciones

    posiciones_vistas = set()  # evitar duplicados por overlap
    for chunk, offset in _dividir_en_chunks(texto):
        doc = nlp(chunk)
        for token in doc:
            lema = token.lemma_.lower()
            if not token.is_stop and not token.is_punct and lema.strip():
                pos_inicio = token.idx + offset
                pos_fin = pos_inicio + len(token.text)
                if (pos_inicio, pos_fin) in posiciones_vistas:
                    continue
                posiciones_vistas.add((pos_inicio, pos_fin))
                lemas.add(lema)
                if lema not in posiciones:
                    posiciones[lema] = []
                    frecuencias[lema] = 0
                posiciones[lema].append((pos_inicio, pos_fin))
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
    doc = nlp(texto)

    return set([
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ])
