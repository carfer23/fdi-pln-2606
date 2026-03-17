from textual.app import App, ComposeResult
from textual.widgets import Input, Static
from textual.containers import VerticalScroll

from bs4 import BeautifulSoup
import spacy

# Cargar modelo de spaCy
nlp = spacy.load("es_core_news_sm")

QUIJOTE_ASCII = r"""
                  /\       ,,                                        ./
          .---.   ||      /||                                       //
       --'-----`--||    .'  \                                      //
         {{{N `(  ||  .'    @                                     //
         {{{` _/  ||.'    |  \                        _________ _//
         {{{.-.   ||  /  /\   \                        "-------(_)
          {( )| .'||    /  `.  \                               | \\
__        {|\ \'  / )  /     \\O|                              |_|\\
  `-.____.-| \ \ /\/  /       `'                               |_| \\
 -     ////|  \ Y /| |                                         [ ]  \\
   |   |||||`-|\^/|| |                                         F-J   `\
       |||||`-| " [] /                                        J.-'L
     _ \\\\/`-|   []|\                                        ]`-.[
 ) |`---``| _ |__([]| \                                       |.-'|
  /       |/ `|   FJ|\ \                                      [`-.]
 /        `|  |   FJ) \ \                                     F.-'J
/          |  |   FJ|  \ )                                   J`-._ L
|          |  F  J  L  ||                                    ]    >[
`.         )-(> '----` ||                                    | .-' |
`.\        | |    |||  ||                                    [<    ]
| \\       |-|    ||| / |                                    F `-. J
 \ )\    *_)/`-.__|| \\ |                                   J     ` L
"""

def tokenizar_query(texto):
    """Solo para la consulta del usuario. Devuelve un set de lemas."""
    doc = nlp(texto)
    return set([
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ])

def procesar_texto_capitulo(texto):
    """Para indexar el capítulo. Devuelve lemas y las posiciones (índices) en el texto original."""
    doc = nlp(texto)
    lemas = set()
    posiciones = {} # Diccionario: lema -> lista de tuplas (inicio, fin)

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

def buscar(palabra, capitulos):
    resultados = []
    palabras_busqueda = tokenizar_query(palabra)

    if not palabras_busqueda:
        return []

    for cap in capitulos:
        # LÓGICA OR: Intersección de conjuntos. 
        lemas_encontrados = palabras_busqueda & cap["lemas"]
        
        if lemas_encontrados:
            lema_pivote = list(lemas_encontrados)[0]
            idx_inicio, idx_fin = cap["posiciones"][lema_pivote][0]
            
            ventana = 150
            inicio_contexto = max(0, idx_inicio - ventana)
            fin_contexto = min(len(cap["texto"]), idx_fin + ventana)
            
            fragmento_resaltado = cap["texto"][inicio_contexto:fin_contexto]
            
            # 1. Recopilar las palabras exactas que aparecen en este fragmento (sin duplicados)
            palabras_a_resaltar = set()
            for lema in lemas_encontrados:
                for pos_ini, pos_fin in cap["posiciones"][lema]:
                    if pos_ini >= inicio_contexto and pos_fin <= fin_contexto:
                        palabra_original = cap["texto"][pos_ini:pos_fin]
                        palabras_a_resaltar.add(palabra_original)

            # 2. Ordenar las palabras de mayor a menor longitud
            # Esto evita que al reemplazar "viento" rompamos "vientos"
            palabras_a_resaltar = sorted(list(palabras_a_resaltar), key=len, reverse=True)

            # 3. Hacer el reemplazo UNA sola vez por palabra
            for palabra_original in palabras_a_resaltar:
                fragmento_resaltado = fragmento_resaltado.replace(
                    palabra_original, 
                    f"[b green]{palabra_original}[/b green]"
                )

            prefijo = "..." if inicio_contexto > 0 else ""
            sufijo = "..." if fin_contexto < len(cap["texto"]) else ""
            
            texto_final = f"{prefijo}{fragmento_resaltado}{sufijo}"
            
            resultados.append((cap["titulo"], texto_final))

    return resultados


class BuscadorQuijote(App):

    CSS = """
    Screen {
        layout: vertical;
    }

    Input {
        margin: 1;
    }

    #resultados {
        height: 1fr;
        margin: 1;
        border: solid white;
        padding: 1;
    }

    .resultado-item {
        margin-bottom: 2;
        border-bottom: dashed grey;
        padding-bottom: 1;
    }

    .logo {
        text-align: center;
        color: goldenrod;
        margin-bottom: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.capitulos = separar_capitulos()

    def compose(self) -> ComposeResult:
        yield Static(QUIJOTE_ASCII, classes="logo")
        yield Input(placeholder="Buscar palabra... (Ej: los molinos gigantes)", id="busqueda")
        yield VerticalScroll(id="resultados")

    def on_input_submitted(self, event: Input.Submitted):
        palabra = event.value.strip()

        contenedor = self.query_one("#resultados")
        contenedor.remove_children()

        resultados = buscar(palabra, self.capitulos)

        if not resultados:
            contenedor.mount(Static("No se encontraron resultados"))
            return

        for titulo, contexto in resultados:
            # Creamos un bloque para cada resultado
            texto = f"[b yellow]{titulo}[/b yellow]\n\n{contexto}"
            contenedor.mount(Static(texto, classes="resultado-item"))


if __name__ == "__main__":
    BuscadorQuijote().run()