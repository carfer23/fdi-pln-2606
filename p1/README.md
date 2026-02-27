# Práctica 1 - Agente de IA

En esta práctica se implementa un agente de IA que es capaz de gestionar recursos e intercambiarlos con otros agentes para llegar a acumular los recursos objetivo definidos.

FALTA:
- explicacion breve de cada .py
- explicar mas o menos que hace el agente
- cómo clonar el repositorio?
- instrucciones para ejecutar wheel

## Configuración

### Ollama
La documentación e instalación completa de Ollama puede encontrarse [aquí](https://ollama.com/). Para instalarlo, se pueden seguir los siguientes pasos:
```
wget https://ollama.com/download/ollama-linux-amd64.tar.zst

tar -xvf ollama-linux-amd64.tar.zst
```
Para iniciar Ollama, se utiliza el siguiente comando:
```
./bin/ollama serve
```
Es necesario dejar esta terminal con Ollama iniciado para poder ejecutar el agente.

### Variables de entorno
En primer lugar, puede ser necesario modificar el archivo ``.env`` para ajustar los valores de las variables de entorno. Por defecto se usan los siguientes:
```
# Configuración del servidor
FDI_PLN__BUTLER_ADDRESS=http://147.96.84.134:7719/

# Configuración del agente
AGENT_NAME=trilobite

# Configuración de Ollama
OLLAMA_MODEL=qwen3-vl:8b
```

## Ejecución

### Modo wheel
COMPLETAR

### Modo laboratorio
IMPORTANTE: ejecutar siempre desde el directorio p1/.

Si estamos en el laboratorio, con el resto de agentes conectándose a la misma URL del servidor butler, es suficiente con ejecutar el siguiente comando para iniciar el agente:
```
uv run main.py
```

### Modo local
IMPORTANTE: ejecutar siempre desde el directorio p1/.

Para ejecutar el agente de forma local, es necesario iniciar el servidor butler primero. Para ello debe descargarse uno de los wheel disponibles en el campus. Una vez descargado, la forma más fácil de usarlo es con `uv tool install <ruta al wheel>`, lo que hará disponible en la terminal el comando `fdi-pln-butler`. Para iniciar el servidor, indicamos el siguiente comando en una terminal:
```
fdi-pln-butler server
```
Una vez esté corriendo el servidor, se inicia el agente:
```
uv run main.py
```
Para poder ver el comportamiento del agente, se pueden iniciar varios agentes con disintios alias, cada uno en una terminal. Para ello se puede iniciar el servidor de la siguiente forma:
```
fdi-pln-butler server --monopuesto --buzon
```
Y después iniciar cada agente indicando su alias:
```
AGENT_NAME=agente1 uv run main.py
```