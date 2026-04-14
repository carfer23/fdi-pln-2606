"""Tokenizador BPE (Byte Pair Encoding) mínimo, entrenado sobre el texto"""

from collections import Counter

class BPETokenizer():
    """BPE entrenado sobre un texto.

    Vocabulario inicial: caracteres únicos del texto. Durante el
    entrenamiento se buscan los pares adyacentes más frecuentes y se
    fusionan en nuevos tokens, hasta 'num_merges' veces.
    """
    
    def __init__(self, text, num_merges=500):
        #self.N = N # Tamaño del vocabulario

        chars = sorted(set(text)) # Conjunto de los caracteres/bytes del texto
        self.tok2id = {c: i for i, c in enumerate(chars)} # Diccionario con claves: chars y valores: índice
        self.id2tok = {i: c for c, i in self.tok2id.items()} # Diccionario contrario, claves: índices y valores: chars

        tokens = [self.tok2id[c] for c in text] # Lista con los índices (tokeniza el texto)
        self.merges = [] # Lista de ((id_a, id_b), nuevo_id)
        
        # Se van mergeando los pares más frecuentes
        for _ in range(num_merges):
            pairs = Counter(zip(tokens, tokens[1:])) # Cuenta los pares de tokens más comunes
            if not pairs:
                break
            best = pairs.most_common(1)[0][0] # Pareja más común

            new_id = len(self.tok2id)
            new_tok = self.id2tok[best[0]] + self.id2tok[best[1]]
            self.tok2id[new_tok] = new_id
            self.id2tok[new_id] = new_tok
            self.merges.append(best, new_id)

            # Recorre todo el texto y va aplicando el merge
            tokens = self._apply_merge(tokens, best[0], best[1], new_id)

        self.vocab_size = len(self.tok2id)

@staticmethod
def _apply_merge(tokens, a, b, new_id):
    """Reemplaza todas las ocurrencias del par (a, b) por new_id."""
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return

def encode(self, text):
    tokens = [self.tok2id.get(c, 0) for c in text]

    for (a, b), new_id in self.merges:
        tokens = self._apply_merge(tokens, a, b, new_id)
    return tokens

#def decode

