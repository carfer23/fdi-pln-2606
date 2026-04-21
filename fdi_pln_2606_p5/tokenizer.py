"""Tokenizador BPE (Byte Pair Encoding) mínimo, entrenado sobre el texto."""

from collections import Counter

class BPETokenizer():
    """BPE entrenado sobre un texto.

    Vocabulario inicial: caracteres únicos del texto. Durante el
    entrenamiento se buscan los pares adyacentes más frecuentes y se
    fusionan en nuevos tokens, hasta alcanzar `vocab_size` tokens.

    NOTA: para ser BPE de verdad, tendríamos que hacerlo sobre bytes, no sobre
    caracteres, pero para la práctica funciona bien.
    """
    
    def __init__(self, text, vocab_size=300):
        """Entrena el tokenizador sobre el texto dado.
        
        :param text: texto de entrenamiento.
        :param vocab_size: número de tokens que tendrá el vocabulario final.
        """
        self.vocab_size = vocab_size # Tamaño del vocabulario (número de tokens)

        self.vocab = sorted(set(text)) # Conjunto de los caracteres del texto (sin repetidos)
        self.tok2id = {tok: i for i, tok in enumerate(self.vocab)} # Diccionario con claves: tokens y valores: índice
        #self.id2tok = {i: tok for tok, i in self.tok2id.items()} # Diccionario contrario, claves: índices y valores: tokens

        tokens = [self.tok2id[c] for c in text] # Lista con los índices de cada token (el texto tokenizado)
        self.merges = [] # Lista de ((id_a, id_b), nuevo_id), para encode()
        
        # Se van mergeando los pares más frecuentes hasta alcanzar el tamaño del vocabulario
        for new_id in range(len(self.vocab), vocab_size):
            pairs = Counter(zip(tokens, tokens[1:])) # Cuenta los pares de tokens más comunes
            best = pairs.most_common(1)[0][0] # Pareja más común

            # El nuevo token es la concatenación de los tokens de la pareja más común
            new_tok = self.vocab[best[0]] + self.vocab[best[1]]

            self.tok2id[new_tok] = new_id
            self.vocab.append(new_tok)
            self.merges.append((best, new_id)) # ((id_a, id_b), nuevo_id)

            # Recorre todo el texto y va aplicando el merge
            tokens = self._apply_merge(tokens, best[0], best[1], new_id)

    @staticmethod
    def _apply_merge(tokens, a, b, new_id):
        """Reemplaza todas las ocurrencias del par (a, b) por new_id."""
        result = []
        i = 0
        # Recorre toda la lista de tokens del texto
        while i < len(tokens):
            # Si encuentra el par (a, b) en el texto, lo reemplaza por new_id y avanza dos posiciones
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                result.append(new_id)
                i += 2
            # Si no encuentra el par, añade el token actual a result y avanza una posición
            else:
                result.append(tokens[i])
                i += 1
        return result

    def encode(self, text):
        """Codifica un texto aplicando los merges aprendidos."""
        # Si un carácter no está en el vocabulario, se asigna el token 0 por defecto
        tokens = [self.tok2id.get(tok, 0) for tok in text]

        # Aplica los merges aprendidos en el entrenamiento por orden
        for (a, b), new_id in self.merges:
            tokens = self._apply_merge(tokens, a, b, new_id)

        return tokens

    def decode(self, ids):
        """Decodifica una lista de ids a texto."""
        # Obtiene los tokens asociados a cada id
        text = [self.vocab[id] for id in ids]

        # Concatena los tokens directamente (BPE ya incluye los espacios como parte del token)
        return "".join(text)

    def __repr__(self):
        pretty = [t.replace("\n", "\\n").replace(" ", "▁") for t in self.vocab]
        return f"{len(self.vocab)} tokens: ['{"', '".join(pretty)}']"

# Si ejecutamos este módulo directamente, probamos el tokenizador
if __name__ == "__main__":
    import sys
    from pathlib import Path

    files_path = Path(sys.argv[1] if len(sys.argv) > 1 else "resources")
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    textos = "\n\n".join(open(p).read() for p in files_path.glob("*.txt"))
    tokenizer = BPETokenizer(textos, vocab_size=vocab_size)
    print(tokenizer)
