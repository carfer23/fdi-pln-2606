from pathlib import Path
import argparse
import re
import typer

app = typer.Typer()

# Mapeo de acentos
ACCENT_MAP = {
    "a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú",
    "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú",
}

# Diéresis
UMLAUT_MAP = {"u": "ü", "U": "Ü"}

# Vocales
VOWELS = set("aeiouAEIOU")

# Cierra paréntesis antes de estos signos
PAREN_CLOSE_BEFORE = {"."}

# Números codificados: i=0, j=1, ... r=9
DIGIT_LETTERS = "ijklmnopqr"
LETTER_TO_DIGIT = {ch: str(idx) for idx, ch in enumerate(DIGIT_LETTERS)}

# Token numérico (deben ser minúsculas)
NUMTOKEN_RE = re.compile(r"[i-r]+(?:\.[i-r]+)*")

def caesar_bytes(data: bytes, k: int) -> bytes:
    """
    Aplica un desplazamiento César a una secuencia de bytes.
    
    :param data: Secuencia de bytes a transformar.
    :param k: Desplazamiento a aplicar.
    :return: Nueva secuencia de bytes con el desplazamiento aplicado.
    """
    return bytes((b + k) & 0xFF for b in data)

def apply_to_previous(out_chars: list[str], mapping: dict):
    """Aplica una transformación a la última cadena del buffer, si existe."""
    if out_chars:
        out_chars[-1] = mapping.get(out_chars[-1], out_chars[-1])

def strip_trailing_space(out: list[str]):
    """Elimina espacios al final del buffer (para evitar ' )')."""
    while out:
        last = out[-1]
        if last == " ":
            out.pop()
            continue
        if isinstance(last, str) and last.endswith(" "):
            out[-1] = last.rstrip(" ")
            if out[-1] == "":
                out.pop()
            continue
        break

def close_paren_if_open(out: list[str], in_paren: bool) -> bool:
    """Si hay un paréntesis abierto, lo cierra."""
    if in_paren:
        strip_trailing_space(out)
        if not out or out[-1] != ")":
            out.append(")")
        return False
    return in_paren

def open_paren(out: list[str]) -> None:
    """Abre paréntesis con un espacio antes si hace falta: 'Nombre (Club'."""
    if out and out[-1] not in (" ", "\n") and not out[-1].endswith((" ", "(", "«", "—")):
        out.append(" ")
    out.append("(")

def decode_numeric_tokens(text: str) -> str:
    """Decodifica tokens numéricos."""
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        out = []
        for ch in tok:
            if ch == ".":
                out.append(".")
            else:
                out.append(LETTER_TO_DIGIT[ch])
        return "".join(out)

    return NUMTOKEN_RE.sub(repl, text)

def clean_decoded_text(decoded: str) -> str:
    """Aplica las reglas de transformación al texto decodificado."""

    # Saltos de línea
    decoded = decoded.replace("77", "\n\n")
    decoded = decoded.replace("7", "\n")

    # Inicio de texto o tras \n\n) empieza con 8 (espacio) -> cursiva (NO FUNCIONA)
    #decoded = re.sub(r"(^|\n\n)8(.*?)(?=\n\n|$)", r"\1\033[1m\2\033[0m", decoded, flags=re.DOTALL)

    # Espacios
    decoded = decoded.replace("8", " ")

    # Decodifica tokens numéricos 
    decoded = decode_numeric_tokens(decoded)

    out: list[str] = []
    i = 0
    n = len(decoded)

    dash_open = True   # para {{
    quote_open = True  # para }
    in_paren = False   # paréntesis abierto por ~

    while i < n:
        ch = decoded[i]

        # Si hay puntuación fuerte y hay paréntesis abierto, se cierra
        if in_paren and (ch in PAREN_CLOSE_BEFORE or decoded[i:i+2] == "\n\n"):
            in_paren = close_paren_if_open(out, in_paren)

        # {{ -> raya (toggle)
        if ch == "{" and i + 1 < n and decoded[i + 1] == "{":
            out.append(" —" if dash_open else "— ")
            dash_open = not dash_open
            i += 2
            continue

        # } -> « » (toggle)
        if ch == "}":
            out.append("«" if quote_open else "»")
            quote_open = not quote_open
            i += 1
            continue

        # ~ -> abre paréntesis; si ya había uno abierto, lo cerramos antes (sin espacio extra)
        if ch == "~":
            if in_paren:
                in_paren = close_paren_if_open(out, in_paren)
            open_paren(out)
            in_paren = True
            i += 1
            continue

        # | -> ’
        if ch == "|":
            out.append("’")
            i += 1
            continue

        # s -> punto (.) solo si no es parte de un número (antes de i..r o M)
        if ch == "s":
            if in_paren:
                in_paren = close_paren_if_open(out, in_paren)
            out.append(".")
            i += 1
            continue

        # t -> coma (,)
        if ch == "t":
            if i + 1 == n or decoded[i + 1] in (" ", "\n"):
                if in_paren:
                    in_paren = close_paren_if_open(out, in_paren)
                out.append(",")
            else:
                out.append("t")
            i += 1
            continue

        # u -> punto y coma (;)
        if ch == "u":
            if in_paren:
                in_paren = close_paren_if_open(out, in_paren)
            out.append(";")
            i += 1
            continue

        # v -> dos puntos (:)
        if ch == "v":
            if in_paren:
                in_paren = close_paren_if_open(out, in_paren)
            out.append(":")
            i += 1
            continue

        # _ -> tilde en la vocal anterior
        if ch == "_":
            apply_to_previous(out, ACCENT_MAP)
            i += 1
            continue

        # ` -> diéresis en la vocal anterior
        if ch == "`":
            apply_to_previous(out, UMLAUT_MAP)
            i += 1
            continue

        # b -> mayúscula en la anterior
        if ch == "b":
            if out:
                out[-1] = out[-1].upper()
            i += 1
            continue

        # a tras n + vocal -> ñ; si no, a normal.
        # Esto se hace aquí para que el marcador 'a' no interfiera con otras reglas (como acentos o mayúsculas)
        if ch == "a":
            if out and out[-1].lower() == "n" and (i + 1 < n) and (decoded[i + 1] in VOWELS):
                out[-1] = "Ñ" if out[-1].isupper() else "ñ"
                i += 1  # consume solo el marcador
                continue
            out.append("a")
            i += 1
            continue

        # Resto de letras -> minúscula
        if ch.isalpha():
            out.append(ch.lower())
        else:
            out.append(ch)
        i += 1

    # Si queda un paréntesis abierto al final, se cierra
    if in_paren:
        _ = close_paren_if_open(out, in_paren)

    text = "".join(out)

    # Normaliza espacios múltiples
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Ajuste de espacios con rayas
    text = re.sub(r"\s+—\s*", " —", text)
    text = re.sub(r"—\s+", "— ", text)

    return text

@app.command()
def decode(fichero: Path, k: int = 45):
    """Decodifica un fichero de PLNCG26 a UTF8."""
    if not fichero.exists():
        typer.echo(f"Error: El archivo {fichero} no existe.", err=True)
        raise typer.Exit(1)
    
    # Lee el archivo como bytes
    data = fichero.read_bytes()

    # Aplica el desplazamiento César
    plain_bytes = caesar_bytes(data, k)

    # Decodifica usando latin-1 para preservar los bytes
    decoded_raw = plain_bytes.decode("latin-1", errors="ignore")

    # Aplica las reglas de transformación al texto decodificado
    final_text = clean_decoded_text(decoded_raw)
    
    typer.echo(final_text)

@app.command()
def encode(fichero: Path, k: int = 45):
    """Codifica un fichero de UTF8 a PLNCG26 (No implementado)."""
    typer.echo("Operación 'encode' no implementada todavía.")

def main():
    app()

if __name__ == "__main__":
    main()