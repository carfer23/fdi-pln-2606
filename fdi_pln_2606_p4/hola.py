from textual.app import App, ComposeResult
from textual.widgets import Input, Static
from textual.containers import VerticalScroll

import re
from bs4 import BeautifulSoup
import re
from collections import Counter

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
--'--'"""""`------''--'`'""""""""""""""""""""""""""""""""""""""""""""""
"""

with open("2000-h.htm", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

capitulos = []  # aquí cargas tus capítulos
for h3 in soup.find_all("h3"):
    a = h3.find("a")

    if a and a.get("name"):
        nombre = a["name"]

        # solo capítulos
        if "_" in nombre:
            
            titulo = h3.get_text(strip=True)

            texto = []
            nodo = h3.find_next_sibling()

            while nodo and nodo.name != "h3":
                if nodo.name == "p":
                    texto.append(nodo.get_text(" ", strip=True))
                nodo = nodo.find_next_sibling()

            capitulos.append({
                "id": nombre,
                "titulo": titulo,
                "texto": " ".join(texto)
            })

def buscar(palabra):
    resultados = []

    patron = r"\b" + re.escape(palabra.lower()) + r"\b"

    for cap in capitulos:
        parrafos = cap["texto"].split("\n")

        for p in parrafos:
            if re.search(patron, p.lower()):
                resultados.append((cap["titulo"], p))

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
    }

    .logo {
    text-align: center;
    color: goldenrod;
    margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(QUIJOTE_ASCII, classes="logo")
        yield Input(placeholder="Buscar palabra...", id="busqueda")
        yield VerticalScroll(id="resultados")

    def on_input_submitted(self, event: Input.Submitted):

        palabra = event.value.strip()

        contenedor = self.query_one("#resultados")
        contenedor.remove_children()

        resultados = buscar(palabra)

        if not resultados:
            contenedor.mount(Static("No se encontraron resultados"))
            return

        for titulo, parrafo in resultados:
            texto = f"[{titulo}]\n{parrafo}\n"
            contenedor.mount(Static(texto))


if __name__ == "__main__":
    BuscadorQuijote().run()