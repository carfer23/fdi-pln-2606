from bs4 import BeautifulSoup
import spacy

# Cargar modelo de spaCy
nlp = spacy.load("es_core_news_sm")

def procesar_texto_capitulo(texto):
    """Indexa un capítulo. Devuelve lemas y las posiciones (índices) en el texto original."""
    # Procesamos el texto con spaCy para obtener lemas y sus posiciones
    doc = nlp(texto)

    lemas = set()
    posiciones = {} # Diccionario: lema -> lista de tuplas (inicio, fin)

    # Recorremos cada token para extraer lemas y sus posiciones
    for token in doc:
        lema = token.lemma_.lower()
        if not token.is_stop and not token.is_punct and lema.strip():
            lemas.add(lema)
            if lema not in posiciones:
                posiciones[lema] = []
            # Guardamos dónde empieza y dónde acaba la palabra original
            posiciones[lema].append((token.idx, token.idx + len(token.text)))
            
    return lemas, posiciones

def separar_capitulos():
    """Lee el HTML del Quijote, separa los capítulos y devuelve una lista de diccionarios con 
    id, título, texto completo, lemas y posiciones."""
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
                
                # Procesamos y extraemos posiciones
                lemas, posiciones = procesar_texto_capitulo(texto_completo)
                
                capitulos.append({
                    "id": nombre,
                    "titulo": titulo,
                    "texto": texto_completo,
                    "lemas": lemas,
                    "posiciones": posiciones
                })

    return capitulos

def tokenizar_query(texto):
    """Solo para la consulta del usuario. Devuelve un set de lemas."""
    # Procesamos la consulta con spaCy
    doc = nlp(texto)

    return set([
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ])

def busqueda_clasica(palabra, capitulos):
    """Busca la palabra (o palabras) en los capítulos. Devuelve una lista de tuplas (título, fragmento)."""
    resultados = []
    palabras_busqueda = tokenizar_query(palabra)

    if not palabras_busqueda:
        return []

    for cap in capitulos:
        # Intersección de conjuntos. 
        lemas_encontrados = palabras_busqueda & cap["lemas"]
        
        if lemas_encontrados:
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
            resultados.append((cap["titulo"], texto_final))

    return resultados
