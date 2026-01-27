import requests
import sys

url_base = "http://147.96.81.252:8000"

def add_alias(nombre):
    url_alias = f"{url_base}/alias/{nombre}"
    r = requests.post(url_alias)
    return r

def main(nombre):
    if len(sys.argv) != 2:
        print("Uso: python script.py <alias>")
        sys.exit(1)

    nombre = sys.argv[1]
    r = add_alias(nombre)

    if r.status_code == 200:
        print(f"Alias '{nombre}' registrado!")
    else:
        print("No se ha podido registrar")
        print("Status:", r.status_code)
        print("Respuesta:", r.text)


if __name__ == "__main__":
    main()
