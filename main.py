from fastapi import FastAPI
import requests

app = FastAPI()
url_base = "http://147.96.81.252:8000"

@app.post("/register/{nombre}")
def add_alias(nombre: str):
    url_alias = f"{url_base}/alias/{nombre}"
    r = requests.post(url_alias)
    if r.status_code == 200:
        return {"message": f"Alias '{nombre}' registrado!"}
    else:
        return {
            "message": "No se ha podido registrar",
            "status": r.status_code,
            "respuesta": r.text
        }

