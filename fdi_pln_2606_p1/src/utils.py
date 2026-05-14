"""Módulo con utilidades generales para el sistema."""

import sys
import re
from loguru import logger
from rich.console import Console
from rich.theme import Theme
from config import AGENT_NAME

# Console compartida para toda la aplicación
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
})
console = Console(theme=custom_theme)
original_stdout = sys.stdout

class StreamToLogger:
    """Clase para redirigir la salida de los print a Loguru."""
    def __init__(self, level="INFO"):
        self.level = level
        # Expresión regular para quitar los códigos ANSI de rich
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, buffer):
        original_stdout.write(buffer) # Imprime en la consola con colores preservados
        for line in buffer.rstrip().splitlines():
            # Eliminamos los colores para el archivo log
            clean_line = self.ansi_escape.sub('', line.rstrip())
            if clean_line:
                logger.opt(depth=1).log(self.level, clean_line)

    def flush(self):
        original_stdout.flush()

def setup_logger():
    """Configura Loguru para escribir en un archivo (sin códigos ANSI)."""
    logger.remove() # Eliminamos el handler por defecto
    
    # Handler solo para el archivo. La consola la manejará Rich directamente.
    logger.add(f"logs/agente_{AGENT_NAME}.log", format="{time:YYYY-MM-DD HH:mm:ss} | {message}", encoding="utf-8")

    # Redirigimos sys.stdout a Loguru para que todo lo que imprima rich vaya al fichero
    sys.stdout = StreamToLogger("INFO")
