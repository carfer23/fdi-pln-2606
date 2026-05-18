# Práctica 1 - Agente de IA

En esta práctica se implementa un agente de IA que es capaz de gestionar recursos e intercambiarlos con otros agentes para llegar a acumular los recursos objetivo definidos.

## Integrantes
- Carmen Fernández González
- Yushan Yang Xu

## NUEVO: Resumen de mejoras para la reentrega

### Ingeniería de contexto
* **Prompts inteligentes:** los prompts se adaptan a la situación de la partida:
  * Prioridad máxima si está a pocos recursos de ganar.
  * Ajustes estratégicos si le sobran o faltan recursos en exceso.
  * Bloqueo de tratos si se ha quedado sin excedentes.
  * *Prompt constraints* si detecta el uso de modelos SLM (pequeños), mitigando sus alucinaciones.
* **Corrección semántica:** ajuste de los prompts ya existentes para arreglar la confusión recurrente del LLM entre los conceptos "enviar" y "recibir".
* **Prevención de bucles:** corrección para no mandar cartas confirmando la recepción de un envío cuando es respuesta a otra confirmación.

### Código
* **Tipado y docstrings:** cabeceras de funciones tipadas y documentadas.
* **Modularidad:** creación de una clase `AgenteBase` reutilizable y personalizable, facilitando el despliegue de agentes.

### Funcionalidad
* **Estructuras de datos:** introducción de `Pydantic` y dataclasses (modelos específicos para las Cartas y el Estado de los Recursos) para garantizar la consistencia de los datos.
* **Mejora de la terminal:** interfaz rediseñada usando la librería `rich`. Se usan tablas y paneles a color, para hacer la información más legible.
* **Registro de logs:** sistema de logging con `loguru` que registra la trazabilidad en ficheros.

## Requisitos

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
Puede ser necesario modificar el archivo ``.env`` para ajustar los valores de las variables de entorno. Por defecto se usan los siguientes:
```
# Configuración del servidor
FDI_PLN__BUTLER_ADDRESS=http://147.96.84.134:7719/

# Configuración del agente
AGENT_NAME=trilobites

# Configuración de Ollama
OLLAMA_MODEL=qwen2.5:3b
```

## Ejecución

### Con wheel
El wheel se encuentra disponible en la *release* del repositorio. Instalar el wheel y ejecutarlo con los siguientes comandos:
```
uv tool install <ruta al wheel>
uv run fdi-pln-2606-p1
```

### En el laboratorio
Si estamos en el laboratorio, con el resto de agentes conectándose a la misma URL del servidor butler, es suficiente con ejecutar el siguiente comando para iniciar el agente:
```
uv run main.py
```

### En local
Para ejecutar el agente en local, es necesario iniciar el servidor butler primero. Para ello, debe descargarse uno de los wheel disponibles en el campus. Una vez descargado, se puede instalar con `uv tool install <ruta al wheel>`, lo que hará disponible en la terminal el comando `fdi-pln-butler`.

Para poder ver el comportamiento del agente, se pueden iniciar varios agentes con disintios alias, cada uno en una terminal. Para ello se puede iniciar el servidor de la siguiente forma:
```
fdi-pln-butler server --monopuesto --buzon
```
Y después iniciar cada agente indicando su alias:
```
AGENT_NAME=agente1 uv run main.py
```
