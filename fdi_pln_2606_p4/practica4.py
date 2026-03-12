from bs4 import BeautifulSoup
import re
from collections import Counter

with open("2000-h.htm", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

capitulos = []

for h3 in soup.find_all("h3"):
    a = h3.find("a")

    if a and a.get("name"):
        nombre = a["name"]

        # solo capítulos
        if "_" in nombre:
            
            titulo = h3.get_text(strip=True)

            texto = []
            nodo = h3.find_next_sibling()

            while nodo and nodo.name != "h3":
                if nodo.name == "p":
                    texto.append(nodo.get_text(" ", strip=True))
                nodo = nodo.find_next_sibling()

            capitulos.append({
                "id": nombre,
                "titulo": titulo,
                "texto": " ".join(texto)
            })

stopwords = {
    "el","la","los","las","de","del","que","y","a","en","un","una",
    "por","con","para","al","se","su","sus","lo","como","más"
}

# def palabras_importantes(texto):
#     texto = texto.lower()
#     palabras = re.findall(r"\b[a-záéíóúñ]+\b", texto)
#     palabras = [p for p in palabras if p not in stopwords and len(p) > 3]
#     conteo = Counter(palabras)
#     return conteo.most_common(10)

# for cap in capitulos:
#     print(cap["titulo"])
#     print(palabras_importantes(cap["texto"]))
#     print()
#     break

def buscar(capitulo_lista, consulta):

    consulta = consulta.lower()

    for cap in capitulo_lista:
        if consulta in cap["texto"].lower():

            print("\n======================")
            print(cap["titulo"])
            print("======================")

            parrafos = cap["texto"].split("\n")

            for p in parrafos:
                if consulta in p.lower():
                    print(p)
                    print()

while True:

    consulta = input("\nBuscar palabra (o 'salir'): ")

    if consulta == "salir":
        break

    buscar(capitulos, consulta)