import json
import math
import os
import re
import unicodedata
import urllib.error
import urllib.request
from collections import Counter

from bs4 import BeautifulSoup

# Textual no renderiza bien si TERM llega como "dumb".
if os.environ.get("TERM", "").lower() in {"", "dumb"}:
    os.environ["TERM"] = "xterm-256color"

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Select, Static

MODELO_EMBEDDINGS = "nomic-embed-text"
MODELO_CHAT = "llama3"
OLLAMA_URL = "http://localhost:11434"
CACHE_FILE = "embeddings_cache.json"

TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")
STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
    "para", "con", "no", "una", "su", "al", "lo", "como", "mas", "pero", "sus", "le",
    "ya", "o", "este", "si", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre",
    "tambien", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante",
    "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
}

QUIJOTE_ASCII = r"""
                  /\       ,,                                        ./
          .---.   ||      /||                                       //
       --'-----`--||    .'  \                                      //
         {{{N `(  ||  .'    @                                     //
         {{{` _/  ||.'    |  \                        _________ _//
         {{{.-.   ||  /  /\   \                        "-------(_)
          {( )| .'||    /  `.  \                               | \\
__        {|\ \'  / )  /     \\O|                              |_|\ \
  `-.____.-| \ \ /\/  /       `'                               |_| \
 -     ////|  \ Y /| |                                         [ ]  \
   |   |||||`-|\^/|| |                                         F-J   `\
       |||||`-| " [] /                                        J.-'L
     _ \\\\/`-|   []|\                                        ]`-.[
 ) |`---``| _ |__([]| \                                       |.-'|
  /       |/ `|   FJ|\ \                                      [`-.]
 /        `|  |   FJ) \ \                                     F.-'J
/          |  |   FJ|  \ )                                   J`-._ L
|          |  F  J  L  ||                                    ]    >[
`.         )-(> '----` ||                                    | .-' |
`.\        | |    |||  ||                                    [<    ]
| \\       |-|    ||| / |                                    F `-. J
 \ )\    *_)/`-.__|| \\ |                                   J     ` L
"""


def _normalizar_token(token):
    token = token.lower()
    token = unicodedata.normalize("NFD", token)
    token = "".join(ch for ch in token if unicodedata.category(ch) != "Mn")
    return token


def _tokenizar(texto):
    tokens = []
    for token in TOKEN_RE.findall(texto):
        token_n = _normalizar_token(token)
        if len(token_n) > 2 and token_n not in STOPWORDS:
            tokens.append(token_n)
    return tokens


def separar_capitulos():
    """Lee el HTML y preprocesa tokens/frecuencias para búsqueda clásica."""
    try:
        with open("2000-h.htm", "r", encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        return []

    soup = BeautifulSoup(html, "html.parser")
    capitulos = []

    for h3 in soup.find_all("h3"):
        a = h3.find("a")
        if not (a and a.get("name") and "_" in a["name"]):
            continue

        titulo = h3.get_text(strip=True)
        texto_nodos = []
        nodo = h3.find_next_sibling()
        while nodo and nodo.name != "h3":
            if nodo.name == "p":
                texto_nodos.append(nodo.get_text(" ", strip=True))
            nodo = nodo.find_next_sibling()

        texto_completo = " ".join(texto_nodos)
        tokens = _tokenizar(texto_completo)
        frecuencias = Counter(tokens)

        capitulos.append({
            "titulo": titulo,
            "texto": texto_completo,
            "lemas": set(frecuencias.keys()),
            "frecuencias": dict(frecuencias),
        })

    return capitulos


def _ollama_disponible():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2):
            return True
    except Exception:
        return False


def _get_embedding(texto):
    """Llama a Ollama para obtener el vector del texto."""
    data = json.dumps({"model": MODELO_EMBEDDINGS, "input": texto[:3000]}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read())

    if "embeddings" not in payload or not payload["embeddings"]:
        raise RuntimeError("Ollama no devolvió embeddings válidos.")

    return payload["embeddings"][0]


def _chat_ollama(prompt):
    data = json.dumps(
        {
            "model": MODELO_CHAT,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read())

    contenido = payload.get("message", {}).get("content")
    if not contenido:
        raise RuntimeError("Ollama no devolvió una respuesta válida para RAG.")

    return contenido


def _similitud_coseno(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class BuscadorQuijote(App):
    CSS = """
    Screen { layout: vertical; background: #1a1a1a; }
    .logo { text-align: center; color: goldenrod; margin: 1; }
    #controles { height: auto; align: center middle; }
    Input { margin: 1; border: double goldenrod; }
    #resultados { height: 1fr; margin: 1; border: solid gray; padding: 1; }
    .resultado-item { margin-bottom: 1; border-bottom: dashed white 20%; padding: 1; }
    """

    def on_mount(self):
        self.query_one("#busqueda").focus()
        self.run_worker(self.inicializar_datos())

    async def inicializar_datos(self):
        contenedor = self.query_one("#resultados")
        contenedor.remove_children()
        contenedor.mount(Static("⏳ Cargando capítulos de El Quijote..."))

        self.capitulos = separar_capitulos()
        self.embeddings_listos = False

        if not self.capitulos:
            contenedor.remove_children()
            contenedor.mount(Static("❌ No se pudo leer '2000-h.htm'."))
            return

        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    embeddings = json.load(f)

                if len(embeddings) == len(self.capitulos):
                    for cap, emb in zip(self.capitulos, embeddings):
                        cap["embedding"] = emb
                    self.embeddings_listos = True
                    contenedor.mount(Static("📂 Embeddings cargados desde caché."))
                else:
                    contenedor.mount(
                        Static("⚠️ Caché de embeddings desactualizada; se regenerará al usar modo semántico.")
                    )
            except Exception:
                contenedor.mount(Static("⚠️ No se pudo leer la caché; se regenerará al usar modo semántico."))

        if not self.embeddings_listos:
            contenedor.mount(Static("ℹ️ Los modos semántico y RAG generarán embeddings solo al primer uso."))

        contenedor.mount(Static("✅ ¡Listo! Introduce tu búsqueda arriba."))

    async def asegurar_embeddings(self):
        if self.embeddings_listos:
            return True

        contenedor = self.query_one("#resultados")

        if not _ollama_disponible():
            contenedor.mount(
                Static("❌ Ollama no está activo en localhost:11434. Ejecuta `ollama serve` y reintenta.")
            )
            return False

        contenedor.mount(Static("🤖 Generando embeddings (solo la primera vez)..."))
        todos_embs = []

        try:
            total = len(self.capitulos)
            for i, cap in enumerate(self.capitulos, start=1):
                emb = _get_embedding(cap["texto"])
                cap["embedding"] = emb
                todos_embs.append(emb)

                if i % 10 == 0 or i == total:
                    contenedor.mount(Static(f"   > Procesados {i}/{total} capítulos..."))

            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(todos_embs, f)

            self.embeddings_listos = True
            return True
        except (urllib.error.URLError, TimeoutError) as e:
            contenedor.mount(Static(f"❌ Error conectando con Ollama: {e}"))
            return False
        except Exception as e:
            contenedor.mount(Static(f"❌ Error generando embeddings: {e}"))
            return False

    def compose(self) -> ComposeResult:
        yield Static(QUIJOTE_ASCII, classes="logo")
        yield Horizontal(
            Select(
                [
                    ("Clásica (TF)", "clasica"),
                    ("Semántica (IA)", "semantica"),
                    ("RAG (respuesta)", "rag"),
                ],
                value="clasica",
                id="modo",
            ),
            Button("Salir", id="btn_salir", variant="error"),
            id="controles",
        )
        yield Input(placeholder="Ej: los molinos de viento...", id="busqueda")
        yield VerticalScroll(id="resultados")

    async def on_input_submitted(self, event: Input.Submitted):
        query = event.value.strip()
        if not query:
            return

        modo = self.query_one("#modo").value
        nombre_modo = {
            "clasica": "clásico",
            "semantica": "semántico",
            "rag": "rag",
        }.get(modo, str(modo))
        contenedor = self.query_one("#resultados")
        contenedor.remove_children()
        contenedor.mount(Static(f"🔍 Buscando '{query}' en modo {nombre_modo}..."))
        self.run_worker(self.procesar_busqueda(query, modo))

    async def procesar_busqueda(self, query, modo):
        contenedor = self.query_one("#resultados")
        resultados = []

        try:
            if modo == "clasica":
                tokens_q = set(_tokenizar(query))
                for cap in self.capitulos:
                    score = sum(cap["frecuencias"].get(token, 0) for token in tokens_q)
                    if score > 0:
                        resultados.append((score, cap))
                resultados.sort(key=lambda x: x[0], reverse=True)

            elif modo == "semantica":
                ok = await self.asegurar_embeddings()
                if not ok:
                    return

                emb_q = _get_embedding(query)
                for cap in self.capitulos:
                    sim = _similitud_coseno(emb_q, cap["embedding"])
                    resultados.append((sim, cap))
                resultados.sort(key=lambda x: x[0], reverse=True)

            elif modo == "rag":
                ok = await self.asegurar_embeddings()
                if not ok:
                    return

                emb_q = _get_embedding(query)
                ranking_semantico = []
                for cap in self.capitulos:
                    sim = _similitud_coseno(emb_q, cap["embedding"])
                    ranking_semantico.append((sim, cap))
                ranking_semantico.sort(key=lambda x: x[0], reverse=True)

                tokens_q = set(_tokenizar(query))
                ranking_clasico = []
                for cap in self.capitulos:
                    score = sum(cap["frecuencias"].get(token, 0) for token in tokens_q)
                    if score > 0:
                        ranking_clasico.append((score, cap))
                ranking_clasico.sort(key=lambda x: x[0], reverse=True)

                capitulos_contexto = []
                vistos = set()
                for _, cap in ranking_semantico[:3]:
                    if cap["titulo"] not in vistos:
                        vistos.add(cap["titulo"])
                        capitulos_contexto.append(cap)
                for _, cap in ranking_clasico[:3]:
                    if cap["titulo"] not in vistos:
                        vistos.add(cap["titulo"])
                        capitulos_contexto.append(cap)

                if not capitulos_contexto:
                    contenedor.remove_children()
                    contenedor.mount(Static("❌ No se encontró contexto para responder."))
                    return

                contexto = "\n\n---\n\n".join(
                    f"[{cap['titulo']}]\n{cap['texto'][:900]}" for cap in capitulos_contexto
                )
                prompt = (
                    "Eres un experto en El Quijote. Responde en español usando solo los pasajes.\n"
                    "Si no está en los pasajes, dilo claramente.\n"
                    "Al final añade 'Fuentes:' con los títulos usados.\n\n"
                    f"PASAJES:\n{contexto}\n\n"
                    f"PREGUNTA: {query}\n\nRESPUESTA:"
                )

                respuesta = _chat_ollama(prompt)
                contenedor.remove_children()
                contenedor.mount(
                    Static(
                        f"[b yellow]Respuesta RAG[/b yellow]\n\n{respuesta}",
                        classes="resultado-item",
                    )
                )
                return

            else:
                contenedor.remove_children()
                contenedor.mount(Static(f"❌ Modo no implementado: {modo}"))
                return

            contenedor.remove_children()
            if not resultados:
                contenedor.mount(Static("❌ No se encontraron resultados."))
            else:
                for score, cap in resultados[:5]:
                    texto_final = (
                        f"[b yellow]{cap['titulo']}[/b yellow] (Relevancia: {score:.2f})\n\n"
                        f"{cap['texto'][:300]}..."
                    )
                    contenedor.mount(Static(texto_final, classes="resultado-item"))
        except Exception as e:
            contenedor.mount(Static(f"[b red]Error:[/b red] {e}"))

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn_salir":
            self.exit()


if __name__ == "__main__":
    BuscadorQuijote().run()
