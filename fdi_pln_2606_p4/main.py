from textual.app import App, ComposeResult
from textual.widgets import Input, Static, Select, Button
from textual.containers import VerticalScroll, Horizontal

from busqueda_clasica import separar_capitulos, busqueda_clasica

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


class BuscadorQuijote(App):

    CSS = """
    Screen {
        layout: vertical;
    }

    Input, Select, Button {
        margin: 1;
    }

    #controles {
        height: auto;
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
        # Cargamos y procesamos los capítulos al iniciar la aplicación
        self.capitulos = separar_capitulos()

    def compose(self) -> ComposeResult:
        yield Static(QUIJOTE_ASCII, classes="logo")
        yield Horizontal(
            Select(
                [("Búsqueda clásica", "clasica")],
                value="clasica",
                id="modo_operacion"
            ),
            Button("Salir", id="btn_salir", variant="error"),
            id="controles"
        )
        yield Input(placeholder="Introducir consulta...", id="busqueda")
        yield VerticalScroll(id="resultados")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn_salir":
            self.exit()

    def on_input_submitted(self, event: Input.Submitted):
        palabra = event.value.strip()
        modo = self.query_one("#modo_operacion", Select).value

        contenedor = self.query_one("#resultados")
        contenedor.remove_children()

        if modo == "clasica":
            # Realizamos la búsqueda clásica
            resultados = busqueda_clasica(palabra, self.capitulos)

            if not resultados:
                contenedor.mount(Static("No se encontraron resultados"))
                return

            for titulo, contexto in resultados:
                # Creamos un bloque para cada resultado
                texto = f"[b yellow]{titulo}[/b yellow]\n\n{contexto}"
                contenedor.mount(Static(texto, classes="resultado-item"))
        else:
            contenedor.mount(Static(f"Modo '{modo}' no implementado todavía.", classes="resultado-item"))


if __name__ == "__main__":
    BuscadorQuijote().run()