"""Módulo que maneja la interacción con Ollama."""

import ollama
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config import OLLAMA_MODEL
from utils import console

P1_DIR = Path(__file__).resolve().parent


# Definimos el formato de la respuesta esperada de Ollama usando Pydantic
class DecisionAgente(BaseModel):
    razonamiento: str
    accion: Literal["enviar_paquete", "esperar"]
    destinatario: str = Field(description="Alias del agente destinatario, si la acción es enviar_paquete")
    recurso_enviar: str = Field(description="Nombre del recurso a enviar, si la acción es enviar_paquete")
    cantidad_recurso_enviar: int = Field(description="Cantidad del recurso a enviar, si la acción es enviar_paquete")
    recurso_recibir: str = Field(description="Nombre del recurso a recibir, si la acción es recibir_paquete")
    cantidad_recurso_recibir: int = Field(description="Cantidad del recurso a recibir, si la acción es recibir_paquete")


def generar_contexto_adaptativo(faltantes: dict, sobrantes: dict, as_system: bool = False) -> str:
    """Genera instrucciones adaptando la estrategia según el estado de la partida.
    
    :param faltantes: dict de recursos que faltan para el objetivo
    :param sobrantes: dict de recursos en exceso que se pueden ofrecer
    :param as_system: Si True, el texto se formateará como instrucciones para el sistema; si False, se añadirá como contexto adicional para el usuario. Esto afecta principalmente al título
    :return: Texto con instrucciones adaptativas para el agente
    """
    instrucciones = []
    
    total_faltantes = sum(faltantes.values())
    total_sobrantes = sum(sobrantes.values())

    if total_faltantes == 1:
        instrucciones.append("SITUACIÓN CRÍTICA: Estás a una sola unidad de ganar. Tienes prioridad máxima para conseguir ese recurso faltante. Sé agresivo.")
    elif total_faltantes > 5:
        instrucciones.append("FASE TEMPRANA: Te faltan muchos recursos. Prioriza conseguir una diversidad de objetos sin gastar demasiados sobrantes de golpe.")

    if total_sobrantes == 0:
        instrucciones.append("RECURSOS AGOTADOS: No tienes nada en 'SOBRANTES' para intercambiar. Debes RECHAZAR/ESPERAR en cualquier trato que te exija dar recursos.")
    elif total_sobrantes > 5:
        instrucciones.append("ABUNDANCIA: Tienes muchos recursos sobrantes. Puedes permitirte tratos donde des algo que te sobra en gran cantidad a cambio de lo que necesitas.")

    if not instrucciones:
        return ""

    titulo = "ADAPTACIÓN A LA SITUACIÓN ACTUAL" if not as_system else "INSTRUCCIONES ESTRATÉGICAS"
    contenido = "\n- ".join(instrucciones)
    return f"\n\n--- {titulo} ---\n- {contenido}\n"


def generar_contexto_llm() -> str:
    """Genera advertencias específicas según las fortalezas o debilidades del LLM."""
    modelo_lower = OLLAMA_MODEL.lower()
    instrucciones = []

    # Modelos pequeños (SLMs) suelen sufrir al ceñirse al formato JSON y a veces alucinan objetos
    if any(m in modelo_lower for m in ["1b", "3b", "2b", "mini"]):
        instrucciones.append("⚠️ DEBILIDAD DEL MODELO DETECTADA: Debido al tamaño de la IA en uso, TIENES TENDENCIA A ALUCINAR NOMBRES DE RECURSOS u olvidar tu inventario. REVISA TRES VECES que el 'recurso_enviar' está EXACTAMENTE ESCRITO dentro de SOBRANTES.")
        instrucciones.append("Solo la acción 'enviar_paquete' requiere llenar los demás campos, si es 'esperar' déjalos por defecto.")
    
    if "llama3" in modelo_lower or "qwen" in modelo_lower:
        instrucciones.append("TIP DE MODELO: Eres muy bueno razonando paso a paso. Úsalo a tu favor detallando todo el proceso lógico en el campo 'razonamiento' EXPLICANDO QUIÉN DA QUÉ Y A QUIÉN, antes de generar la decisión.")

    if not instrucciones:
        return ""

    return "\n\n--- ADVERTENCIAS DEL METAMODELO ---\n- " + "\n- ".join(instrucciones) + "\n"

def cargar_prompt(nombre_archivo: str, **kwargs) -> str:
    """Carga y formatea un prompt desde archivo, y le añade inteligencia adaptativa.
    
    :param nombre_archivo: Nombre del archivo de prompt (sin extensión)
    :param kwargs: Variables para formatear el prompt
    :return: Prompt formateado listo para usar
    """
    path = P1_DIR / "prompts" / f"{nombre_archivo}.txt"
    try:
        texto_base = path.read_text(encoding="utf-8").format(**kwargs)
    except FileNotFoundError:
        console.print(f"[error]❌ Prompt no encontrado: {path}[/error]")
        return ""
    except KeyError as exc:
        console.print(f"[error]❌ Variables faltantes en prompt {path}: {exc}[/error]")
        return ""

    # Inyección de adaptabilidad
    if "faltantes" in kwargs and "sobrantes" in kwargs:
        texto_base += generar_contexto_adaptativo(kwargs["faltantes"], kwargs["sobrantes"])
    
    texto_base += generar_contexto_llm()

    return texto_base


def ollama_generate(prompt: str, system_prompt: str = "") -> dict:
    """Consulta a Ollama y devuelve la decisión del agente como diccionario.
    
    :param prompt: Prompt a enviar a Ollama
    :param system_prompt: Prompt del sistema
    :return: Decisión del agente como diccionario
    """
    if not OLLAMA_MODEL:
        console.print("[error]❌ OLLAMA_MODEL no configurado. Se omite consulta.[/error]")
        return {"accion": "esperar"}

    if not prompt:
        console.print("[warning]⚠️ Prompt vacío. Se omite consulta.[/warning]")
        return {"accion": "esperar"}

    try:
        with console.status("[bold cyan]🤖 Analizando situación en Ollama...[/bold cyan]", spinner="dots"):
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                format=DecisionAgente.model_json_schema(),
                options={"temperature": 0.1},
            )
        
        decision = DecisionAgente.model_validate_json(response["message"]["content"])
        result = decision.model_dump()
        
        # Formateado bonito del resultado de la IA
        accion = result.get('accion')
        color = "green" if accion == "enviar_paquete" else "yellow"
        
        console.print(f"[bold {color}]📋 Decisión: {accion}[/bold {color}]")
        console.print(f"[dim italic]   → Razón: {result.get('razonamiento', '')[:200]}...[/dim italic]")
        
        return result

    except Exception as e:
        console.print(f"[error]⚠️ Error Ollama:[/error] {e}")
        return {"accion": "esperar"}
