from pathlib import Path
import argparse
import re

ACCENT_MAP = {
    "a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú",
    "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú",
}
UMLAUT_MAP = {"u": "ü", "U": "Ü"}
VOWELS = set("aeiouAEIOU")

# Cierra paréntesis antes de estos signos
PAREN_CLOSE_BEFORE = {".", ",", ";", ":"}

# Dígitos codificados: i=0, j=1, ... r=9
DIGIT_LETTERS = "ijklmnopqr"
LETTER_TO_DIGIT = {ch: str(idx) for idx, ch in enumerate(DIGIT_LETTERS)}  # i->0 ... r->9

# Token numérico: opcional "M" + letras i..r + opcional "."
NUMTOKEN_RE = re.compile(r"\bM?[i-r]+(?:\.[i-r]+)*\b", re.IGNORECASE)

def caesar_bytes(data: bytes, k: int, mode: str) -> bytes:
    if mode == "plus":
        return bytes((b + k) & 0xFF for b in data)
    if mode == "minus":
        return bytes((b - k) & 0xFF for b in data)
    raise ValueError("mode debe ser 'plus' o 'minus'")

def apply_to_previous(out_chars: list[str], mapping: dict):
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
    """Convierte tokens tipo jp, jo.li, Mjp, km usando i..r -> 0..9."""
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        prefix = ""
        core = tok
        if core and core[0] in ("M", "m"):
            prefix = core[0].upper()
            core = core[1:]
        out = []
        for ch in core:
            if ch == ".":
                out.append(".")
            else:
                out.append(LETTER_TO_DIGIT.get(ch.lower(), ch))
        return prefix + "".join(out)

    return NUMTOKEN_RE.sub(repl, text)

def clean_decoded_text(decoded: str) -> str:
    out: list[str] = []
    i = 0
    n = len(decoded)

    dash_open = True   # para {{
    quote_open = True  # para }
    in_paren = False   # paréntesis abierto por ~

    while i < n:
        ch = decoded[i]

        # Si viene puntuación fuerte y estamos dentro de paréntesis, cerramos antes del signo
        if ch in PAREN_CLOSE_BEFORE and in_paren:
            in_paren = close_paren_if_open(out, in_paren)
            out.append(ch)
            i += 1
            continue

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

        # ~ -> abre paréntesis; si ya había uno abierto, lo cerramos ANTES (sin espacio extra)
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

        # Espacios
        if ch in ("8", "7"):
            out.append(" ")
            i += 1
            continue

        # Punto (s)
        if ch == "s":
            if in_paren:
                in_paren = close_paren_if_open(out, in_paren)
            out.append(".")
            i += 1
            continue

        # Coma (t) solo si final de palabra (antes de 8/7 o fin)
        if ch == "t":
            if i + 1 == n or decoded[i + 1] in ("8", "7"):
                if in_paren:
                    in_paren = close_paren_if_open(out, in_paren)
                out.append(",")
            else:
                out.append("t")
            i += 1
            continue

        # Punto y coma (u) -> ;
        if ch == "u":
            if in_paren:
                in_paren = close_paren_if_open(out, in_paren)
            out.append(";")
            i += 1
            continue

        # Dos puntos (v) -> :
        if ch == "v":
            if in_paren:
                in_paren = close_paren_if_open(out, in_paren)
            out.append(":")
            i += 1
            continue

        # Tilde
        if ch == "_":
            apply_to_previous(out, ACCENT_MAP)
            i += 1
            continue

        # Diéresis
        if ch == "`":
            apply_to_previous(out, UMLAUT_MAP)
            i += 1
            continue

        # Mayúscula en la anterior
        if ch == "b":
            if out:
                out[-1] = out[-1].upper()
            i += 1
            continue

        # Regla Ñ: 'a' marcador tras n si luego viene vocal
        if ch == "a":
            if out and out[-1].lower() == "n" and (i + 1 < n) and (decoded[i + 1] in VOWELS):
                out[-1] = "Ñ" if out[-1].isupper() else "ñ"
                i += 1  # consume solo el marcador
                continue
            out.append("a")
            i += 1
            continue

        # Letras normales -> minúscula
        if ch.isalpha():
            out.append(ch.lower())
        else:
            out.append(ch)
        i += 1

    # Si quedó un paréntesis abierto al final, ciérralo
    if in_paren:
        _ = close_paren_if_open(out, in_paren)

    text = "".join(out)

    # Normaliza espacios múltiples
    text = " ".join(text.split())

    # Ajuste suave de espacios con rayas
    text = re.sub(r"\s+—\s*", " —", text)
    text = re.sub(r"—\s+", "— ", text)

    # Decodifica tokens numéricos (jp -> 17, jo.li -> 16.30, etc.)
    text = decode_numeric_tokens(text)

    return text

def main():
    ap = argparse.ArgumentParser(description="Descifra (César) y limpia principal.bin con reglas personalizadas.")
    ap.add_argument("-i", "--input", default="principal.bin")
    ap.add_argument("-k", type=int, default=45)
    ap.add_argument("--mode", choices=["plus", "minus"], default="plus")
    ap.add_argument("-o", "--output", default="principal_limpio.txt")
    args = ap.parse_args()

    data = Path(args.input).read_bytes()
    plain_bytes = caesar_bytes(data, args.k, args.mode)
    decoded = plain_bytes.decode("latin-1")

    cleaned = clean_decoded_text(decoded)
    Path(args.output).write_text(cleaned + "\n", encoding="utf-8")
    print(cleaned)

if __name__ == "__main__":
    main()