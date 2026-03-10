from pathlib import Path
import re
import typer

app = typer.Typer()

# Mapeo de acentos
ACCENT_MAP_BYT = {
    b"a": b"\xe1", b"e": b"\xe9", b"i": b"\xed", b"o": b"\xf3", b"u": b"\xfa",
    b"A": b"\xc1", b"E": b"\xc9", b"I": b"\xcd", b"O": b"\xd3", b"U": b"\xda",
}

# Diéresis
UMLAUT_MAP_BYT = {b"u": b"\xfc", b"U": b"\xdc"}

# Vocales
VOWELS_BYT = set(b"aeiouAEIOU")

# Códigos ASCII clave para las operaciones con bytes
C_OPEN_BRACE = 123; C_CLOSE_BRACE = 125; C_TILDE = 126; C_PIPE = 124
C_S = 115; C_T = 116; C_U = 117; C_V = 118; C_UNDERSCORE = 95
C_BACKTICK = 96; C_B = 98; C_A = 97; C_N = 110

# Números codificados: i=0, j=1, ... r=9
DIGIT_LETTERS_BYT = b"ijklmnopqr"
LETTER_TO_DIGIT_BYT = {ch: str(idx).encode("ascii") for idx, ch in enumerate(DIGIT_LETTERS_BYT)}

def caesar_bytes(data: bytes, k: int) -> bytes:
    """
    Aplica un desplazamiento César a una secuencia de bytes.
    
    :param data: Secuencia de bytes a transformar.
    :param k: Desplazamiento a aplicar.
    :return: Nueva secuencia de bytes con el desplazamiento aplicado.
    """
    return bytes((b + k) & 0xFF for b in data)

def clean_decoded_bytes(data: bytes) -> bytes:
    """Aplica las reglas de transformación al texto decodificado."""
    
    # Saltos de párrafo y de línea
    data = data.replace(b"77", b"\n\n").replace(b"7", b"\n")
    
    # Inicio de texto o tras \n\n) empieza con 8 (espacio) -> negrita (**)
    data = re.sub(rb"(^|\n\n)8(.*?)(?=\n\n|$)", rb"\1**\2**", data, flags=re.DOTALL)
    
    # Espacios
    data = data.replace(b"8", b" ")
    
    # Decodifica tokens numéricos 
    def repl_num(m: re.Match) -> bytes:
        return b"".join(LETTER_TO_DIGIT_BYT[b] for b in m.group(0))
    data = re.sub(rb"[i-r]+", repl_num, data)

    out = bytearray()
    i, n = 0, len(data)
    
    dash_open = True   # para {{
    quote_open = True  # para }
    in_paren = False   # paréntesis abierto por ~

    while i < n:
        b = data[i]

        # Si hay puntuación fuerte (s, t, u, v) y hay paréntesis abierto, se cierra
        if in_paren and (b in (C_S, C_T, C_U, C_V) or data[i:i+2] == b"\n\n"):
            if out and out[-1] == ord(" "): out.pop()
            out.extend(b")")
            in_paren = False

        # {{ -> raya (toggle)
        if b == C_OPEN_BRACE and i + 1 < n and data[i+1] == C_OPEN_BRACE:
            if dash_open:
                if out and out[-1] != ord(" ") and out[-1] != ord("\n"):
                    out.extend(b" ")
                out.extend(b"\x2d\x2d")
            else:
                out.extend(b"\x2d\x2d")
                # Solo añadimos espacio pos-raya si el siguiente char no es puntuación ni salto de línea
                if i + 2 < n and data[i+2] not in (C_S, C_T, C_U, C_V, ord(" "), ord("\n")):
                    out.extend(b" ")
            dash_open = not dash_open
            i += 2
            continue

        # } -> « » (toggle)
        elif b == C_CLOSE_BRACE:
            out.extend(b"\xab" if quote_open else b"\xbb") # « » en Latin-1
            quote_open = not quote_open
            i += 1
            continue
            
        # ~ -> abre paréntesis; si ya había uno abierto, lo cerramos antes
        elif b == C_TILDE:
            if in_paren: 
                if out and out[-1] == ord(" "): out.pop()
                out.extend(b")")
            if out and out[-1] not in (ord(" "), ord("\n")) and not out.endswith((b"(", b"\xab", b"\x2d\x2d ")):
                out.extend(b" ")
            out.extend(b"(")
            in_paren = True
            i += 1
            continue
            
        # | -> ’
        elif b == C_PIPE: 
            out.extend(b"\x27")
            i += 1
            continue 
            
        # s -> punto (.) 
        elif b == C_S: 
            out.extend(b".")
            i += 1
            continue

        # t -> coma (,) 
        elif b == C_T:
            if i + 1 == n or data[i+1] in (32, 10): 
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
                last = bytes([out[-1]])
                out[-1] = ACCENT_MAP_BYT.get(last, last)[0]
            i += 1
            continue

        # ` -> diéresis en la vocal anterior
        elif b == C_BACKTICK:
            if out:
                last = bytes([out[-1]])
                out[-1] = UMLAUT_MAP_BYT.get(last, last)[0]
            i += 1
            continue
            
        # b -> mayúscula en la anterior
        elif b == C_B:
            if out:
                char = bytes([out[-1]]).decode("latin-1").upper().encode("latin-1")
                out[-1] = char[0]
            i += 1
            continue
            
        # a tras n + vocal -> ñ; si no, a normal.
        elif b == C_A:
            if out and out[-1] in (C_N, ord("N")) and i+1 < n and data[i+1] in VOWELS_BYT:
                is_up = out[-1] == ord("N")
                out[-1] = 0xD1 if is_up else 0xF1 # Ñ o ñ en Latin-1
                i += 1
                continue
            out.append(b)
            i += 1
            continue

        # Resto de letras -> minúscula
        if ord("A") <= b <= ord("Z"): 
            out.append(b + 32)
        else: 
            out.append(b)
        i += 1

    # Si queda un paréntesis abierto al final, se cierra
    if in_paren:
        if out and out[-1] == ord(" "): out.pop()
        out.extend(b")")

    return bytes(out)

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

    final_bytes = clean_decoded_bytes(plain_bytes)
    result_utf8 = final_bytes.decode("latin-1").replace("--", "—")
    
    typer.echo(result_utf8)

@app.command()
def encode(fichero: Path, k: int = 45):
    """Codifica un fichero de UTF8 a PLNCG26 (No implementado)."""
    typer.echo("Operación 'encode' no implementada todavía.")

def main():
    app()

if __name__ == "__main__":
    main()