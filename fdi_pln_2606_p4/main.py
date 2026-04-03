import json
import os
import urllib.request

# # Textual no renderiza bien si TERM llega como "dumb".
# if os.environ.get("TERM", "").lower() in {"", "dumb"}:
#     os.environ["TERM"] = "xterm-256color"

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Select, Static

from utils import separar_capitulos
from busqueda_clasica import busqueda_clasica
from busqueda_semantica import busqueda_semantica, generar_embeddings
from busqueda_rag import busqueda_rag

OLLAMA_URL = "http://localhost:11434"
CACHE_FILE = "embeddings_cache.json"

QUIJOTE_ASCII = r"""
                  /\       ,,                                        ./
          .---.   ||      /||                                       //
       --'-----`--||    .'  \                                      //
         {{{N `(  ||  .'    @                                     //
         {{{` _/  ||.'    |  \                        _________ _//
         {{{.-.   ||  /  /\   \                        "-------(_)
          {( )| .'||    /  `.  \                               | \\
__        {|\ \'  / )  /     \\O|                              |_|\ \
  `-.____.-| \ \ /\/  /       `'                               |_| \
 -     ////|  \ Y /| |                                         [ ]  \
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


def _ollama_disponible():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2):
            return True
    except Exception:
        return False


class BuscadorQuijote(App):
    CSS = """
    Screen { layout: vertical; background: #1a1a1a; }
    Input, Select,  Button { margin: 1; }
    #controles { height: auto; align: center middle; }
    #resultados { height: 1fr; margin: 1; border: solid white; padding: 1; }
    .resultado-item { margin-bottom: 1; border-bottom: dashed white 20%; padding: 1; }
    .logo { text-align: center; color: goldenrod; margin: 1; }
    """

    def on_mount(self):
        self.query_one("#busqueda").focus()
        self.run_worker(self.inicializar_datos())

    async def inicializar_datos(self):
        contenedor = self.query_one("#resultados")
        contenedor.remove_children()

        self.capitulos = separar_capitulos()
        self.embeddings_listos = False

        if not self.capitulos:
            contenedor.remove_children()
            contenedor.mount(Static("❌ No se pudo leer el texto de El Quijote."))
            return

        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    embeddings = json.load(f)

                if len(embeddings) == len(self.capitulos):
                    for cap, emb in zip(self.capitulos, embeddings):
                        cap["embedding"] = emb
                    self.embeddings_listos = True
                    contenedor.mount(Static("📂 Embeddings cargados desde caché."))
                else:
                    contenedor.mount(
                        Static("⚠️ Caché de embeddings desactualizada; se regenerará al usar modo semántico.")
                    )
            except Exception:
                contenedor.mount(Static("⚠️ No se pudo leer la caché; se regenerará al usar modo semántico."))

        contenedor.mount(Static("✅ Aplicación iniciada correctamente."))

    async def asegurar_embeddings(self):
        """Genera o carga los embeddings necesarios para la búsqueda semántica. Devuelve True si están listos."""
        if self.embeddings_listos:
            return True

        contenedor = self.query_one("#resultados")
        contenedor.mount(Static("🤖 Generando embeddings con spaCy..."))

        try:
            embeddings = generar_embeddings(self.capitulos)

            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(embeddings, f)

            self.embeddings_listos = True
            return True
        except Exception as e:
            contenedor.mount(Static(f"❌ Error generando embeddings: {e}"))
            return False

    def compose(self) -> ComposeResult:
        """Construye la interfaz de usuario con Textual."""
        #yield Static(QUIJOTE_ASCII, classes="logo")
        yield Horizontal(
            Select(
                [
                    ("Búsqueda clásica", "clasica"),
                    ("Búsqueda semántica", "semantica"),
                    ("RAG", "rag"),
                ],
                value="clasica",
                id="modo",
            ),
            Button("Salir", id="btn_salir", variant="error"),
            id="controles",
        )
        yield Input(placeholder="Introducir consulta...", id="busqueda")
        yield VerticalScroll(id="resultados")

    async def on_input_submitted(self, event: Input.Submitted):
        query = event.value.strip()
        if not query:
            return

        modo = self.query_one("#modo").value

        contenedor = self.query_one("#resultados")
        contenedor.remove_children()

        self.run_worker(self.procesar_busqueda(query, modo))

    async def procesar_busqueda(self, query, modo):
        contenedor = self.query_one("#resultados")
        resultados = []

        try:
            if modo == "clasica":
                # Búsqueda clásica
                resultados = busqueda_clasica(query, self.capitulos)

                if not resultados:
                    contenedor.mount(Static("No se encontraron resultados"))
                    return

                for score, titulo, contexto in resultados:
                    # Se crea un bloque para cada resultado
                    texto = f"[b yellow]{titulo}[/b yellow] (Relevancia: {score:.4f})\n\n{contexto}"
                    contenedor.mount(Static(texto, classes="resultado-item"))
                return

            elif modo == "semantica":
                ok = await self.asegurar_embeddings()
                if not ok:
                    return

                # Busqueda semántica
                resultados_sem = busqueda_semantica(query, self.capitulos)
                
                if not resultados_sem:
                    contenedor.mount(Static("No se encontraron resultados."))
                    return
                
                # Se muestra el título y fragmento de cada resultado
                for porcentaje, titulo, fragmento in resultados_sem:
                    texto_final = f"[b yellow]{titulo}[/b yellow] (Similitud: {porcentaje:.1f}%)\n\n{fragmento}"
                    contenedor.mount(Static(texto_final, classes="resultado-item"))
                return

            elif modo == "rag":
                if not _ollama_disponible():
                    contenedor.mount(Static("❌ Ollama no está activo en localhost:11434. Necesario para RAG."))
                    return
                
                ok = await self.asegurar_embeddings()
                if not ok:
                    return

                # Búsqueda RAG
                respuesta = busqueda_rag(query, self.capitulos)

                if not respuesta:
                    contenedor.mount(Static("No se encontraron resultados."))
                    return

                contenedor.mount(
                    Static(
                        f"[b yellow]Respuesta RAG[/b yellow]\n\n{respuesta}",
                        classes="resultado-item",
                    )
                )
                return

        except Exception as e:
            contenedor.mount(Static(f"[b red]Error:[/b red] {e}"))

    def on_button_pressed(self, event: Button.Pressed):
        """Botón para salir de la aplicación."""
        if event.button.id == "btn_salir":
            self.exit()


if __name__ == "__main__":
    BuscadorQuijote().run()
