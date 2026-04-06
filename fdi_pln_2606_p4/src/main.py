"""
Módulo principal de la aplicación de búsqueda en El Quijote usando Textual.
Define la interfaz de usuario, maneja eventos y coordina las funciones de búsqueda.
"""

import json
import os
import numpy as np
import asyncio
from typing import Any, Dict, List

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Select, Static

from src.utils import separar_capitulos
from src.busqueda_clasica import busqueda_clasica
from src.busqueda_semantica import (
    busqueda_semantica,
    generar_embeddings,
)
from src.busqueda_rag import busqueda_rag
from src.config import EMBEDDINGS_CACHE_FILE
from src.ascii_art import QUIJOTE_ASCII


def _ollama_disponible():
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


class BuscadorQuijote(App):
    """
    Aplicación interactiva de Textual para buscar en el corpus de El Quijote.
    Soporta búsqueda modular: clásica (TF-IDF), semántica (Embeddings) y RAG (LLM).
    """

    CSS = """
    Screen { layout: vertical; background: #1a1a1a; }
    Input, Select,  Button { margin: 1; }
    #controles { height: auto; align: center middle; }
    #resultados { height: 1fr; margin: 1; border: solid white; padding: 1; }
    .resultado-item { margin-bottom: 1; border-bottom: dashed white 20%; padding: 1; }
    .logo { text-align: center; color: goldenrod; margin: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.app_lista: bool = False
        self.embeddings_listos: bool = False
        self.capitulos: List[Dict[str, Any]] = []

    # ==================
    # CONSTRUCCIÓN DE UI
    # ==================

    def compose(self) -> ComposeResult:
        """Construye la jerarquía de componentes de la interfaz de usuario."""
        # yield Static(QUIJOTE_ASCII, classes="logo")
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
            Button("Regenerar embeddings", id="btn_regenerar", variant="warning"),
            Button("Salir", id="btn_salir", variant="error"),
            id="controles",
        )
        yield Input(placeholder="Introducir consulta...", id="busqueda")
        yield VerticalScroll(id="resultados")

    def on_mount(self) -> None:
        """Inicializa los datos y procesos en background al cargar la pantalla."""
        self.query_one("#busqueda").focus()
        self.run_worker(self.inicializar_datos())

    # =============================
    # LÓGICA DE DATOS Y EMBEDDINGS
    # =============================

    async def inicializar_datos(self):
        """Inicializa la aplicación cargando los capítulos, generando o recuperando embeddings y mostrando instrucciones."""
        contenedor = self.query_one("#resultados")
        contenedor.remove_children()

        contenedor.mount(
            Static("📖 Cargando y procesando los capítulos de El Quijote...")
        )
        await asyncio.sleep(0.1)

        try:
            self.capitulos = separar_capitulos()
        except Exception as e:
            contenedor.mount(
                Static(f"❌ Error: No se pudo leer el texto de El Quijote.\n{e}")
            )
            return

        if not self.capitulos:
            contenedor.mount(
                Static("❌ Error: No se encontraron capítulos de El Quijote.")
            )
            return

        self.embeddings_listos = False
        contenedor.mount(Static("🔍 Verificando estado de caché de los embeddings..."))
        await asyncio.sleep(0.1)

        if os.path.exists(EMBEDDINGS_CACHE_FILE):
            try:
                with open(EMBEDDINGS_CACHE_FILE, "r", encoding="utf-8") as f:
                    embeddings_cache = json.load(f)
                if len(embeddings_cache) == len(self.capitulos):
                    for cap, emb in zip(self.capitulos, embeddings_cache):
                        cap["embedding"] = np.array(emb)
                    self.embeddings_listos = True
                    contenedor.mount(
                        Static(
                            "📂 Embeddings recuperados exitosamente desde la caché local."
                        )
                    )
                    await asyncio.sleep(0.5)
            except Exception:
                pass

        if not self.embeddings_listos:
            contenedor.mount(Static("⏳ Generando embeddings.."))
            await asyncio.sleep(
                0.1
            )  # Pausa breve para que se renderice el mensaje anterior
            ok = await self.asegurar_embeddings()
            if not ok:
                return

        # Flujo de inicio completado
        self.mostrar_instrucciones()
        self.app_lista = True

    async def asegurar_embeddings(self):
        """Genera los embeddings necesarios para la búsqueda semántica. Devuelve True si están listos."""
        if self.embeddings_listos:
            return True

        contenedor = self.query_one("#resultados")

        try:
            embeddings = generar_embeddings(self.capitulos)

            with open(EMBEDDINGS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump([emb.tolist() for emb in embeddings], f)

            self.embeddings_listos = True
            return True
        except Exception as e:
            contenedor.mount(Static(f"❌ Error generando embeddings: {e}"))
            return False

    async def regenerar_embeddings(self):
        """Fuerza la regeneración de embeddings sobrescribiendo el archivo cache."""
        contenedor = self.query_one("#resultados")
        contenedor.remove_children()
        contenedor.mount(
            Static("⏳ Regenerando los embeddings forzosamente... por favor, espera.")
        )
        await asyncio.sleep(
            0.1
        )  # Pausa breve para que se renderice el mensaje anterior

        # Desactivamos el flag para asegurar que se llamen de nuevo
        self.embeddings_listos = False

        ok = await self.asegurar_embeddings()
        if ok:
            self.mostrar_instrucciones(
                "[green]✅ Embeddings regenerados y guardados con éxito.[/green]"
            )

    # ============================================
    # MANEJO DE EVENTOS DE INTERFAZ E INTERACCIÓN
    # ============================================

    def mostrar_instrucciones(self, mensaje_extra=""):
        """Muestra las instrucciones de uso en el panel principal."""
        contenedor = self.query_one("#resultados")
        contenedor.remove_children()

        contenido = "✅ [b]Sistema listo para buscar[/b]\n\n"
        if mensaje_extra:
            contenido += f"{mensaje_extra}\n\n"

        contenido += "💡 [i]Instrucciones de uso:[/i]\n"
        contenido += " 1. Selecciona el modo de búsqueda en el menú superior.\n"
        contenido += " 2. Escribe tu consulta en la barra inferior y presiona Enter.\n"
        contenido += " 3. La respuesta se mostrará en este panel."

        contenedor.mount(Static(contenido, classes="resultado-item"))

        # Mostrar ASCII art
        contenedor.mount(Static(QUIJOTE_ASCII, classes="logo"))

    def on_select_changed(self, event: Select.Changed):
        """Limpia los resultados y la búsqueda al cambiar el modo de operación."""
        # Evitar durante la inicialización o re-renderizado
        if not getattr(self, "app_lista", False):
            return

        if event.select.id == "modo":
            # Borrar la consulta actual
            input_busqueda = self.query_one("#busqueda")
            input_busqueda.value = ""

            # Limpiar el panel de resultados y mostrar de nuevo las instrucciones
            self.mostrar_instrucciones()

            # Devolver el foco al input
            input_busqueda.focus()

    def on_button_pressed(self, event: Button.Pressed):
        """Maneja las pulsaciones de los botones de la interfaz."""
        if event.button.id == "btn_salir":
            self.exit()
        elif event.button.id == "btn_regenerar":
            self.run_worker(self.regenerar_embeddings())

    async def on_input_submitted(self, event: Input.Submitted):
        """Maneja la consulta del usuario al presionar Enter en el input de búsqueda."""
        query = event.value.strip()
        if not query:
            return

        modo = self.query_one("#modo").value

        contenedor = self.query_one("#resultados")
        contenedor.remove_children()

        self.run_worker(self.procesar_busqueda(query, modo))

    async def procesar_busqueda(self, query, modo):
        """
        Procesa la consulta del usuario según el modo seleccionado y muestra los resultados.

        :param query: La consulta ingresada por el usuario.
        :param modo: El modo de búsqueda seleccionado ("clasica", "semantica" o "rag").
        """
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
                    contenedor.mount(
                        Static("❌ Ollama no está iniciado. Necesario para RAG.")
                    )
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


def main():
    BuscadorQuijote().run()


if __name__ == "__main__":
    main()
