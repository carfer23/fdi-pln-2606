"""Módulo principal del agente."""

import time
from loguru import logger
from rich.table import Table
from rich.panel import Panel

from info import get_gente, get_buzon, calcular_estado
from acciones import register_agent, cargar_carta, enviar_carta, borrar_carta, ejecutar_accion
from consulta_ollama import ollama_generate, cargar_prompt
from config import AGENT_NAME, validar_config
from models import EstadoRecursos, Carta
from utils import setup_logger, console

# Tiempo en segundos entre envíos de cartas de difusión
TIMEOUT_REDIFUSION = 60


def fase_difusion(estado: EstadoRecursos, usuarios: list[str]) -> None:
    """Fase 1: Envía propuestas de intercambio a todos los agentes disponibles.
    
    :param estado: EstadoRecursos con faltantes y sobrantes
    :param usuarios: lista de alias de los agentes registrados en el servidor
    """
    console.print(Panel.fit("[bold magenta]Fase 1: Enviando cartas de difusión 📨[/bold magenta]"))

    if not estado.faltantes:
        console.print("[warning]No faltan recursos. Saltando difusión.[/warning]")
        return
    
    if not estado.sobrantes:
        console.print("[warning]Sin sobrantes para ofrecer. No se puede proponer intercambio.[/warning]")
        return
    
    # Convertimos los dicts a listas para poder usar índices
    lista_faltantes = list(estado.faltantes.keys())
    lista_sobrantes = list(estado.sobrantes.keys())

    # Excluimos nuestro propio alias de los destinatarios
    destinatarios = [u for u in usuarios if u != AGENT_NAME]

    if not destinatarios:
        console.print("[warning]No hay otros usuarios activos para enviar cartas.[/warning]")
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
                ofr=recurso_ofrecido
            )

            enviar_carta(usuario, "Propuesta de intercambio", cuerpo)
            time.sleep(1)


def _procesar_buzon(buzon: dict[str, dict], prompt_inicial: str) -> None:
    """Procesa todas las cartas del buzón, refrescando el estado entre cada una.
    
    :param buzon: diccionario con las cartas recibidas, con ID como clave
    :param prompt_inicial: prompt del sistema para las consultas a Ollama
    """
    console.print(f"[bold green]📫 Hay {len(buzon)} mensajes en el buzón. Procesando...[/bold green]")

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

        console.print(f"\n[bold blue]📩 Procesando carta de {remitente}[/bold blue] ([cyan]{asunto}[/cyan])")
        console.print(Panel(cuerpo, title="Contenido de la Carta", title_align="left"))

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


def imprimir_estado(estado: EstadoRecursos) -> None:
    """Muestra el estado del agente usando una tabla."""
    table = Table(title=f"📊 Estado del Agente: {AGENT_NAME}", style="cyan")
    table.add_column("Recurso", justify="left", style="white", no_wrap=True)
    table.add_column("Categoría", style="magenta")
    table.add_column("Cantidad", justify="right", style="green")

    for rec, cant in estado.faltantes.items():
        table.add_row(rec, "[bold red]Faltante[/bold red]", str(cant))
    
    table.add_section()

    for rec, cant in estado.sobrantes.items():
        table.add_row(rec, "[bold green]Sobrante[/bold green]", str(cant))

    console.print(table)


def fase_reactiva(prompt_inicial: str) -> None:
    """Fase 2: El agente reacciona a las cartas que llegan a su buzón.
    Cada cierto tiempo envía nuevas cartas de difusión para asegurar propuestas constantes.
    
    :param prompt_inicial: Prompt inicial para las consultas a Ollama
    """
    console.print(Panel.fit("[bold magenta]👁️ Fase 2: Esperando respuestas y paquetes[/bold magenta]"))

    tiempo_ultima_difusion = time.time()

    # Bucle infinito para mantener al agente activo
    while True:
        estado = calcular_estado()

        # Imprimir estado actual
        imprimir_estado(estado)

        # Chequear si se ha cumplido el objetivo
        if estado.objetivo_cumplido:
            console.print(Panel("[bold yellow]🏆 ¡OBJETIVO CUMPLIDO! El agente ha conseguido todos los recursos.[/bold yellow]", border_style="yellow"))
            break

        # Check del temporizador de difusión independiente del buzón
        tiempo_actual = time.time()
        if (tiempo_actual - tiempo_ultima_difusion) >= TIMEOUT_REDIFUSION:
            console.print(f"\n[info]🔄 Han pasado {TIMEOUT_REDIFUSION} segundos. Reenviando cartas de difusión...[/info]")
            usuarios_actualizados = get_gente()
            fase_difusion(estado, usuarios_actualizados)
            # Reiniciamos el temporizador después de enviar las cartas
            tiempo_ultima_difusion = time.time()

        # Leer buzón
        buzon = get_buzon()

        if not buzon:
            console.print("💤 [dim]Buzón vacío. Esperando...[/dim]")
            # Esperamos un poco antes de revisar el buzón de nuevo
            time.sleep(5)

        else:
            _procesar_buzon(buzon, prompt_inicial)


def main() -> None:
    """Función principal que orquesta el agente."""
    errores = validar_config()
    if errores:
        for error in errores:
            console.print(f"[error]❌ Configuración inválida: {error}[/error]")
        raise RuntimeError("Configuración incompleta. Revisa las variables de entorno.")


    console.print("[bold green]🚀 Iniciando agente...[/bold green]")

    # ----- CONFIGURACIÓN INICIAL -------------------------------
    usuarios = get_gente()

    if AGENT_NAME not in usuarios:
        register_agent(AGENT_NAME)

    estado = calcular_estado()
    
    imprimir_estado(estado)

    prompt_inicial = cargar_prompt(
        "prompt_inicial",
        alias=AGENT_NAME,
        faltantes=estado.faltantes,
        sobrantes=estado.sobrantes,
        usuarios=usuarios,
    )

    # ----- EJECUCIÓN DE LAS FASES ------------------------------
    # 1. El agente envía cartas para buscar los recursos que le faltan
    fase_difusion(estado, usuarios)
    # 2. El agente entra en modo reactivo para responder a los mensajes
    fase_reactiva(prompt_inicial)


if __name__ == "__main__":
    setup_logger() # Configuramos el logger para redirigir los print a consola y archivo

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Ejecución detenida por el usuario (Ctrl+C).")
