"""Módulo principal del agente."""

import time
from loguru import logger
from rich.table import Table
from rich.panel import Panel

from agente_base import AgenteBase
from info import get_gente, get_buzon, calcular_estado
from acciones import (
    register_agent,
    cargar_carta,
    enviar_carta,
    borrar_carta,
    ejecutar_accion,
)
from consulta_ollama import ollama_generate, cargar_prompt
from config import AGENT_NAME, validar_config
from models import EstadoRecursos, Carta
from utils import setup_logger, console

# Tiempo en segundos entre envíos de cartas de difusión
TIMEOUT_REDIFUSION = 60


class Agente(AgenteBase):
    """Agente especializado para intercambio de recursos.

    Implementa un bucle reactivo con dos fases: difusión de propuestas y respuesta a mensajes.

    :param AGENT_NAME: Alias del agente, definido en config.py
    :param _timepo_ultima_difusion: Timestamp del último envío de cartas de difusión, para controlar reenvíos periódicos
    """

    def __init__(self) -> None:
        super().__init__()
        self.agent_name = AGENT_NAME
        self._tiempo_ultima_difusion: float | None = None

    # --- Obligatorios ---
    def validate_config(self) -> list[str]:
        """Valida la configuración necesaria para el agente. Devuelve una lista de errores encontrados."""
        return validar_config()

    def build_system_prompt(self) -> str:
        """Construye el prompt del sistema."""
        usuarios = get_gente()
        if AGENT_NAME not in usuarios:
            register_agent(AGENT_NAME)

        estado = calcular_estado()
        self._print_state(estado)

        return cargar_prompt(
            "prompt_inicial",
            alias=AGENT_NAME,
            faltantes=estado.faltantes,
            sobrantes=estado.sobrantes,
            usuarios=usuarios,
        )

    def run_loop(self, system_prompt: str) -> None:
        """Ejecuta el bucle principal del agente, recibiendo el prompt del sistema ya construido."""
        estado = calcular_estado()
        usuarios = get_gente()

        # ----- EJECUCIÓN DE LAS FASES ------------------------------
        # 1. El agente envía cartas para buscar los recursos que le faltan
        self._fase_difusion(estado, usuarios)
        # 2. El agente entra en modo reactivo para responder a los mensajes
        self._fase_reactiva(system_prompt)

    # --- Opcionales ---
    def on_start(self) -> None:
        """Función opcional que se ejecuta al iniciar el agente, antes de construir el prompt."""
        console.print("[bold green]🚀 Iniciando agente...[/bold green]")

    def _print_config_error(self, message: str) -> None:
        """Imprime un mensaje de error relacionado con la configuración del agente."""
        console.print(f"[error]❌ Configuración inválida: {message}[/error]")

    # --- Específicos ---
    def _print_state(self, state: EstadoRecursos) -> None:
        """Imprime el estado actual de recursos faltantes y sobrantes en formato tabla.

        :param state: EstadoRecursos con faltantes y sobrantes
        """
        table = Table(title=f"📊 Estado del Agente: {AGENT_NAME}", style="cyan")
        table.add_column("Recurso", justify="left", style="white", no_wrap=True)
        table.add_column("Categoría", style="magenta")
        table.add_column("Cantidad", justify="right", style="green")

        for rec, cant in state.faltantes.items():
            table.add_row(rec, "[bold red]Faltante[/bold red]", str(cant))

        table.add_section()

        for rec, cant in state.sobrantes.items():
            table.add_row(rec, "[bold green]Sobrante[/bold green]", str(cant))

        console.print(table)

    def _fase_difusion(self, state: EstadoRecursos, users: list[str]) -> None:
        """Fase 1: Envía propuestas de intercambio a todos los agentes disponibles.

        :param estado: EstadoRecursos con faltantes y sobrantes
        :param usuarios: lista de alias de los agentes registrados en el servidor
        """
        console.print(
            Panel.fit(
                "[bold magenta]Fase 1: Enviando cartas de difusión 📨[/bold magenta]"
            )
        )

        if not state.faltantes:
            console.print("[warning]No faltan recursos. Saltando difusión.[/warning]")
            return

        if not state.sobrantes:
            console.print(
                "[warning]Sin sobrantes para ofrecer. No se puede proponer intercambio.[/warning]"
            )
            return

        # Convertimos los dicts a listas para poder usar índices
        lista_faltantes = list(state.faltantes.keys())
        lista_sobrantes = list(state.sobrantes.keys())

        # Excluimos nuestro propio alias de los destinatarios
        destinatarios = [u for u in users if u != AGENT_NAME]

        if not destinatarios:
            console.print(
                "[warning]No hay otros usuarios activos para enviar cartas.[/warning]"
            )
            return

        # Enviamos una carta a cada usuario, rotando los recursos pedidos y ofrecidos
        for usuario in destinatarios:
            for j, recurso_solicitado in enumerate(lista_faltantes):
                recurso_ofrecido = lista_sobrantes[j % len(lista_sobrantes)]

                cuerpo = cargar_carta(
                    "carta_difusion",
                    dest=usuario,
                    alias=AGENT_NAME,
                    req_cant=1,
                    req=recurso_solicitado,
                    ofr_cant=1,
                    ofr=recurso_ofrecido,
                )

                enviar_carta(usuario, "Propuesta de intercambio", cuerpo)
                time.sleep(1)

    def _on_reactive_start(self) -> None:
        """Inicia la fase reactiva."""
        console.print(
            Panel.fit(
                "[bold magenta]👁️ Fase 2: Esperando respuestas y paquetes[/bold magenta]"
            )
        )
        self._tiempo_ultima_difusion = time.time()

    def _on_goal(self) -> None:
        """Se ejecuta cuando se cumple el objetivo del agente."""
        console.print(
            Panel(
                "[bold yellow]🏆 ¡OBJETIVO CUMPLIDO! El agente ha conseguido todos los recursos.[/bold yellow]",
                border_style="yellow",
            )
        )

    def _on_tick(self, state: EstadoRecursos) -> None:
        """Se ejecuta en cada iteración del bucle reactivo, para controlar la redifusión de cartas."""
        if self._tiempo_ultima_difusion is None:
            self._tiempo_ultima_difusion = time.time()

        tiempo_actual = time.time()
        if (tiempo_actual - self._tiempo_ultima_difusion) >= TIMEOUT_REDIFUSION:
            console.print(
                f"\n[info]🔄 Han pasado {TIMEOUT_REDIFUSION} segundos. Reenviando cartas de difusión...[/info]"
            )
            usuarios_actualizados = get_gente()
            self._fase_difusion(state, usuarios_actualizados)
            # Reiniciamos el temporizador después de enviar las cartas
            self._tiempo_ultima_difusion = time.time()

    def _on_empty_inbox(self) -> None:
        """Se ejecuta cuando el buzón está vacío."""
        console.print("💤 [dim]Buzón vacío. Esperando...[/dim]")
        # Esperamos un poco antes de revisar el buzón de nuevo
        time.sleep(5)

    def _procesar_buzon(self, buzon: dict[str, dict], prompt_inicial: str) -> None:
        """Procesa todas las cartas del buzón, refrescando el estado entre cada una.

        :param buzon: diccionario con las cartas recibidas, con ID como clave
        :param prompt_inicial: prompt del sistema para las consultas a Ollama
        """
        console.print(
            f"[bold green]📫 Hay {len(buzon)} mensajes en el buzón. Procesando...[/bold green]"
        )

        cartas = [Carta.from_dict(carta_id, datos) for carta_id, datos in buzon.items()]

        for carta in cartas:
            # Refrescamos el estado para reflejar paquetes ya enviados en este ciclo
            estado = calcular_estado()
            if estado.objetivo_cumplido:
                break

            # Extraemos la información de la carta
            remitente = carta.remitente
            asunto = carta.asunto
            cuerpo = carta.cuerpo

            console.print(
                f"\n[bold blue]📩 Procesando carta de {remitente}[/bold blue] ([cyan]{asunto}[/cyan])"
            )
            console.print(
                Panel(cuerpo, title="Contenido de la Carta", title_align="left")
            )

            if remitente == "Sistema":
                console.print("[info]ℹ️ Carta del sistema recibida.[/info]")
                borrar_carta(carta.id)
                continue

            prompt = cargar_prompt(
                "prompt_procesar_carta",
                usuario=remitente,
                contenido=cuerpo,
                asunto=asunto,
                faltantes=estado.faltantes,
                sobrantes=estado.sobrantes,
            )

            # Consultamos a Ollama para decidir qué hacer con esta carta
            accion = ollama_generate(prompt, prompt_inicial)

            ejecutar_accion(accion, estado, asunto_recibido=asunto)

            # Eliminar la carta procesada
            borrar_carta(carta.id)

            time.sleep(1)

    def _fase_reactiva(self, prompt_inicial: str) -> None:
        """Fase 2: El agente reacciona a las cartas que llegan a su buzón.
        Cada cierto tiempo envía nuevas cartas de difusión para asegurar propuestas constantes.

        :param prompt_inicial: Prompt inicial para las consultas a Ollama
        """
        self._on_reactive_start()

        # Bucle infinito para mantener al agente activo
        while True:
            estado = calcular_estado()

            # Imprimir estado actual
            self._print_state(estado)

            # Chequear si se ha cumplido el objetivo
            if estado.objetivo_cumplido:
                self._on_goal()
                break

            # Check del temporizador de difusión independiente del buzón
            self._on_tick(estado)

            # Leer buzón
            buzon = get_buzon()
            if not buzon:
                self._on_empty_inbox()
            else:
                self._procesar_buzon(buzon, prompt_inicial)


def main() -> None:
    """Función principal que orquesta el agente."""
    Agente().run()


if __name__ == "__main__":
    setup_logger()  # Configuramos el logger para redirigir los print a consola y archivo

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Ejecución detenida por el usuario (Ctrl+C).")
