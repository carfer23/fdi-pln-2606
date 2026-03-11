from pathlib import Path
import re
import typer

app = typer.Typer()


# ----- CONSTANTES ----------------------------------------------------------

# Mapeo de acentos a UTF-8
ACCENT_MAP_UTF8 = {
    ord(b"a"): b"\xc3\xa1", ord(b"e"): b"\xc3\xa9", ord(b"i"): b"\xc3\xad", ord(b"o"): b"\xc3\xb3", ord(b"u"): b"\xc3\xfa",
    ord(b"A"): b"\xc3\x81", ord(b"E"): b"\xc3\x89", ord(b"I"): b"\xc3\x8d", ord(b"O"): b"\xc3\x93", ord(b"U"): b"\xc3\x9a",
}

# Diéresis a UTF-8
UMLAUT_MAP_UTF8 = {ord(b"u"): b"\xc3\xbc", ord(b"U"): b"\xc3\x9c"}

# Vocales
VOWELS_BYT = set(b"aeiouAEIOU")

# Códigos ASCII para operaciones con bytes
C_OPEN_BRACE = 123; C_CLOSE_BRACE = 125 # guión y comillas
C_VIRGULA = 126 # abre paréntesis
C_PIPE = 124 # apóstrofe
C_S = 115; C_T = 116; C_U = 117; C_V = 118 # signos de puntuación
C_UNDERSCORE = 95 # tilde en la vocal anterior
C_BACKTICK = 96 # diéresis
C_B = 98; C_A = 97; C_N = 110
C_DEL = 127

# Constantes en bytes explícitas
B_SPACE = 32
B_NL = 10
B_N_UPPER = 78
B_N_LOWER = 110
B_A_UPPER = 65
B_Z_UPPER = 90
B_A_LOWER = 97
B_Z_LOWER = 122

# Valores UTF-8 a emitir
UTF8_LQUOTE = b"\xc2\xab" # «
UTF8_RQUOTE = b"\xc2\xbb" # »
UTF8_EM_DASH = b"\xe2\x80\x94" # —
UTF8_APOSTROPHE = b"\xe2\x80\x99" # ’
UTF8_NTILDE_UPPER = b"\xc3\x91" # Ñ
UTF8_NTILDE_LOWER = b"\xc3\xb1" # ñ

# Números codificados
DIGIT_LETTERS_BYT = b"ijklmnopqr"
LETTER_TO_DIGIT_BYT = {ch: str(idx).encode("ascii") for idx, ch in enumerate(DIGIT_LETTERS_BYT)}
DIGIT_TO_LETTER_BYT = {str(idx): bytes([ch]) for idx, ch in enumerate(DIGIT_LETTERS_BYT)}

# Alfabeto esperado en la fase intermedia (tras César, antes de limpiar)
ALLOWED_LOWER_PLAIN_BYT = set(b"abijklmnopqrstuv")
CORE_PLAIN_BYTES = (
    set(range(B_A_UPPER, B_Z_UPPER + 1))
    | ALLOWED_LOWER_PLAIN_BYT
    | {C_OPEN_BRACE, C_CLOSE_BRACE, C_VIRGULA, C_PIPE, C_UNDERSCORE, C_BACKTICK, C_DEL, ord(b"7"), ord(b"8")}
)
ACCENT_PREV_BYTES = set(b"AEIOUaeioubB") # letras que pueden llevar tilde + 'b' para mayúscula tras mayúscula


# ----- FUNCIONES ----------------------------------------------------------

def caesar_bytes(data: bytes, k: int) -> bytes:
    """
    Aplica un desplazamiento César a una secuencia de bytes.
    
    :param data: Secuencia de bytes a transformar.
    :param k: Desplazamiento a aplicar.
    :return: Nueva secuencia de bytes con el desplazamiento aplicado.
    """
    return bytes((b + k) & 0xFF for b in data)

def clean_decoded_bytes(data: bytes) -> bytes:
    """
    Aplica las reglas de transformación al texto decodificado.
    
    :param data: Secuencia de bytes decodificada tras aplicar César.
    :return: Secuencia de bytes transformada según las reglas del enunciado.
    """
    # Saltos de párrafo y de línea
    data = data.replace(b"77", b"\n\n").replace(b"7", b"\n")
    
    # Inicio de texto o tras \n\n) empieza con 8 (espacio) -> negrita (NO FUNCIONA)
    #data = re.sub(rb"(^|\n\n)8(.*?)(?=\n\n|$)", rb"\1**\2**", data, flags=re.DOTALL)
    
    # Espacios
    data = data.replace(b"8", b" ")
    
    # Decodifica tokens numéricos 
    def repl_num(m: re.Match) -> bytes:
        return b"".join(LETTER_TO_DIGIT_BYT[b] for b in m.group(0))
    data = re.sub(rb"[i-r]+", repl_num, data)

    out = bytearray()
    i, n = 0, len(data)
    
    dash_open = True   # para --
    quote_open = True  # para « »
    in_paren = False   # paréntesis abierto por ~

    def close_paren() -> None:
        """Limpia ruido antes de cerrar paréntesis."""
        while out and out[-1] == B_SPACE:
            out.pop()
        out.extend(b")")

    while i < n:
        b = data[i]

        # Cerrar paréntesis abierto -> si hay puntuación fuerte (s, t, u, v)
        if in_paren and (b in (C_S, C_T, C_U, C_V) or data[i:i+2] == b"\n\n"):
            close_paren()
            in_paren = False

        # {{ -> guión (toggle)
        if b == C_OPEN_BRACE and i + 1 < n and data[i+1] == C_OPEN_BRACE:
            if dash_open:
                if out and out[-1] != B_SPACE and out[-1] != B_NL:
                    out.extend(b" ")
                out.extend(UTF8_EM_DASH)
            else:
                out.extend(UTF8_EM_DASH)
                # Solo se añade espacio pos-guión si el siguiente char no es puntuación ni salto de línea
                if i + 2 < n and data[i+2] not in (C_S, C_T, C_U, C_V, B_SPACE, B_NL):
                    out.extend(b" ")
            dash_open = not dash_open
            i += 2
            continue

        # } -> « » (toggle)
        elif b == C_CLOSE_BRACE:
            out.extend(UTF8_LQUOTE if quote_open else UTF8_RQUOTE)
            quote_open = not quote_open
            i += 1
            continue
            
        # ~ -> abre paréntesis
        elif b == C_VIRGULA:
            if in_paren: # si ya había uno abierto, se cierra antes
                close_paren()
            if out and out[-1] not in (B_SPACE, B_NL) and not out.endswith((b"(", UTF8_LQUOTE, UTF8_EM_DASH + b" ")):
                out.extend(b" ")
            out.extend(b"(")
            in_paren = True
            i += 1
            continue
            
        # | -> apóstrofe (’)
        elif b == C_PIPE: 
            out.extend(UTF8_APOSTROPHE)
            i += 1
            continue 
            
        # s -> punto (.) 
        elif b == C_S: 
            out.extend(b".")
            i += 1
            continue

        # t -> coma (,) 
        elif b == C_T:
            if i + 1 == n or data[i+1] in (B_SPACE, B_NL): 
                out.extend(b",")
            else: 
                out.append(b)
            i += 1
            continue
            
        # u -> punto y coma (;)
        elif b == C_U: 
            out.extend(b";")
            i += 1
            continue
            
        # v -> dos puntos (:)
        elif b == C_V: 
            out.extend(b":")
            i += 1
            continue
            
        # _ -> tilde en la vocal anterior
        elif b == C_UNDERSCORE:
            if out:
                last_byte = out[-1]
                if last_byte in ACCENT_MAP_UTF8:
                    out[-1:] = ACCENT_MAP_UTF8[last_byte]
            i += 1
            continue

        # ` -> diéresis en la vocal anterior
        elif b == C_BACKTICK:
            if out:
                last_byte = out[-1]
                if last_byte in UMLAUT_MAP_UTF8:
                    out[-1:] = UMLAUT_MAP_UTF8[last_byte]
            i += 1
            continue
            
        # b -> mayúscula en la anterior
        elif b == C_B:
            if out and B_A_LOWER <= out[-1] <= B_Z_LOWER:
                out[-1] -= 32
            i += 1
            continue
            
        # a tras n + vocal -> ñ
        elif b == C_A:
            if out and out[-1] in (B_N_LOWER, B_N_UPPER) and i+1 < n and data[i+1] in VOWELS_BYT:
                is_up = out[-1] == B_N_UPPER
                out[-1:] = UTF8_NTILDE_UPPER if is_up else UTF8_NTILDE_LOWER
                i += 1
                continue
            out.append(b)
            i += 1
            continue

        # Resto de letras -> minúscula
        if B_A_UPPER <= b <= B_Z_UPPER:
            out.append(b + 32)
        else: 
            out.append(b)
        i += 1

    # Si queda un paréntesis abierto al final, se cierra
    if in_paren:
        close_paren()

    return bytes(out)

def encode_text_to_plain_bytes(text: str) -> bytes:
    """
    Convierte texto UTF-8 a la representación de bytes intermedia previa al César.

    :param text: Texto UTF-8 a codificar.
    :return: Secuencia de bytes en formato intermedio para el encoder.
    """
    out = bytearray()
    
    # Mapeo de caracteres especiales a sus tokens en bytes
    char_map = {
        "á": b"A_", "é": b"E_", "í": b"I_", "ó": b"O_", "ú": b"U_",
        "Á": b"Ab_", "É": b"Eb_", "Í": b"Ib_", "Ó": b"Ob_", "Ú": b"Ub_",
        "ü": b"U`", "Ü": b"Ub`",
        "ñ": b"Na", "Ñ": b"Nba",
        ".": b"s", ",": b"t", ";": b"u", ":": b"v",
        "’": b"|", "'": b"|",
        "«": b"}", "»": b"}",
        "(": b"~", ")": b"\x7f",
        " ": b"8", "\n": b"7",
    }
    
    i = 0
    while i < len(text):
        if text[i:i+2] == "\n\n":
            out.extend(b"77")
            i += 2
            continue
            
        ch = text[i]
        
        if ch in char_map:
            out.extend(char_map[ch])
        elif "0" <= ch <= "9":
            out.extend(DIGIT_TO_LETTER_BYT[ch])
        elif "a" <= ch <= "z":
            out.extend(ch.upper().encode("ascii"))
        elif "A" <= ch <= "Z":
            out.extend(ch.encode("ascii"))
            out.extend(b"b")
        elif ch == "—":
            out.extend(b"{{")
        elif ch == "-":
            out.extend(b"{{")
        else:
            try:
                out.extend(ch.encode("utf-8"))
            except UnicodeEncodeError:
                pass
        i += 1

    return bytes(out)

def estimate_plncg26_probability(data: bytes, k: int = 45) -> tuple[float, dict[str, float]]:
    """
    Estima la probabilidad de que un binario siga el formato PLNCG26.

    Se basa en la "forma" del flujo: alfabeto de tokens esperado, 
    coherencia de marcadores y calidad del resultado tras aplicar César + limpieza.

    :param data: Contenido del archivo original (.bin).
    :param k: Desplazamiento César con el que se intentará decodificar.
    :return: probabilidad en [0,1].
    """
    if not data:
        return 0.0

    # Aplica César para obtener la forma intermedia
    plain = caesar_bytes(data, k)
    n = len(plain)

    # Calcula cuántos bytes pertenecen al alfabeto esperado
    core_hits = sum(1 for b in plain if b in CORE_PLAIN_BYTES)
    core_charset_ratio = core_hits / n # valor entre 0 y 1

    # Cuántos bytes de letras minúsculas no permitidas hay (ruido)
    invalid_lower = sum(1 for b in plain if B_A_LOWER <= b <= B_Z_LOWER and b not in ALLOWED_LOWER_PLAIN_BYT)
    invalid_lower_ratio = invalid_lower / n

    marker_total = 0
    marker_valid = 0

    # Recorre el texto para evaluar la validez de los marcadores (b, _, `) según su contexto
    for i, b in enumerate(plain):
        if b == C_B:
            marker_total += 1
            if i > 0 and B_A_UPPER <= plain[i - 1] <= B_Z_UPPER:
                marker_valid += 1
        elif b in (C_UNDERSCORE, C_BACKTICK):
            marker_total += 1
            if i > 0 and plain[i - 1] in ACCENT_PREV_BYTES:
                marker_valid += 1
        elif b == C_A:
            marker_total += 1
            if i > 0 and plain[i - 1] in (B_N_UPPER, B_N_LOWER, C_B):
                marker_valid += 1

    marker_validity = (marker_valid / marker_total) if marker_total else 0.5

    # Evalúa la coherencia de los guiones ({{)
    open_braces = sum(1 for b in plain if b == C_OPEN_BRACE)
    brace_pairs = sum(1 for i in range(n - 1) if plain[i] == C_OPEN_BRACE and plain[i + 1] == C_OPEN_BRACE)
    double_brace_ratio = (2 * brace_pairs / open_braces) if open_braces else 0.5

    # Aplica la limpieza final y evalúa la calidad del resultado UTF-8
    final_bytes = clean_decoded_bytes(plain)
    decoded_text = final_bytes.decode("utf-8", errors="ignore")
    kept_utf8_len = len(decoded_text.encode("utf-8"))
    utf8_keep_ratio = kept_utf8_len / max(1, len(final_bytes))
    printable_ratio = (
        sum(1 for ch in decoded_text if ch.isprintable() or ch in "\n\r\t") / max(1, len(decoded_text))
    )

    # Combina las métricas calculadas para dar la probabilidad final
    raw_score = (
        0.30 * core_charset_ratio
        + 0.15 * (1.0 - invalid_lower_ratio)
        + 0.20 * marker_validity
        + 0.10 * double_brace_ratio
        + 0.15 * utf8_keep_ratio
        + 0.10 * printable_ratio
    )

    # Para ficheros muy cortos, la evidencia es débil: acercamos a 0.5.
    confidence_by_len = min(1.0, n / 120.0)
    probability = 0.5 + (raw_score - 0.5) * confidence_by_len
    probability = max(0.0, min(1.0, probability))

    return probability


# ----- FUNCIONES DECODE Y ENCODE -------------------------------------------------

@app.command()
def decode(fichero: Path, k: int = 45):
    """
    Decodifica un fichero de PLNCG26 a UTF8.
    
    :param fichero: Ruta al archivo de texto PLNCG26 a decodificar.
    :param k: Desplazamiento César a aplicar (por defecto, 45).
    """
    if not fichero.exists():
        typer.echo(f"Error: El archivo {fichero} no existe.", err=True)
        raise typer.Exit(1)
    
    # Lee el archivo como bytes
    data = fichero.read_bytes()

    # Aplica el desplazamiento César
    plain_bytes = caesar_bytes(data, k)

    # Aplica las reglas de transformación para obtener el texto final en bytes, luego decodifica a UTF-8
    final_bytes = clean_decoded_bytes(plain_bytes)
    result_utf8 = final_bytes.decode("utf-8", errors="ignore")
    
    typer.echo(result_utf8)

@app.command()
def encode(fichero: Path, k: int = 45):
    """
    Codifica un fichero UTF-8 a PLNCG26.
    
    :param fichero: Ruta al archivo de texto UTF-8 a codificar.
    :param k: Desplazamiento César a aplicar (por defecto, 45).
    """
    if not fichero.exists():
        typer.echo(f"Error: El archivo {fichero} no existe.", err=True)
        raise typer.Exit(1)

    # Lee el UTF-8 y lo pasa al formato interno previo al cifrado César
    text = fichero.read_text(encoding="utf-8").replace("—", "--")  # Reemplaza em dash por doble guion

    # Convierte el texto a bytes, validando que solo contiene caracteres representables en latin-1.
    try:
        plain_bytes = encode_text_to_plain_bytes(text)
    except ValueError as err:
        typer.echo(f"Error: {err}", err=True)
        raise typer.Exit(1)

    # Aplica el César inverso para producir el formato PLNCG26
    encoded_bytes = caesar_bytes(plain_bytes, -k)

    typer.echo(encoded_bytes)

@app.command()
def detect(fichero: Path, k: int = 45):
    """
    Estima la probabilidad de que un fichero binario sea texto en PLNCG26.

    :param fichero: Ruta al archivo .bin que se quiere analizar.
    :param k: Desplazamiento César a probar (por defecto, 45).
    :param detallar: Si es True, imprime métricas internas.
    """
    if not fichero.exists():
        typer.echo(f"Error: El archivo {fichero} no existe.", err=True)
        raise typer.Exit(1)

    data = fichero.read_bytes()
    probability = estimate_plncg26_probability(data, k)

    typer.echo(f"Probabilidad PLNCG26: {probability * 100:.2f}%")

    if probability >= 0.80:
        typer.echo("Diagnóstico: Muy probable que sea PLNCG26.")
    elif probability >= 0.60:
        typer.echo("Diagnóstico: Probable, pero con incertidumbre.")
    elif probability >= 0.40:
        typer.echo("Diagnóstico: Inconcluso.")
    else:
        typer.echo("Diagnóstico: Poco probable que sea PLNCG26.")

def main():
    app()

if __name__ == "__main__":
    main()
