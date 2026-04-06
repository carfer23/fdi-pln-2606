# Práctica 4 - Buscador Interactivo de El Quijote

## Integrantes
- Carmen Fernández González
- Yushan Yang Xu

## Descripción

En esta práctica se implementa un motor de búsqueda avanzado e interactivo sobre el _corpus_ de la novela *Don Quijote de la Mancha*. 

El objetivo principal de este proyecto es aplicar técnicas modernas de recuperación de información y generación de texto a través de las siguientes características clave:

1. **Preprocesamiento del texto:** El documento HTML original se procesa por capítulos y se divide en fragmentos de texto (*chunks* de 2000 caracteres con 200 caracteres de solapamiento). Durante la indexación, se eliminan las palabras vacías (*stop words*) y se lematizan los términos utilizando `spaCy`.
2. **Interfaz de Usuario en Terminal (TUI):** Se ha construido una interfaz interactiva utilizando `textual`. Esta interfaz permite a los usuarios escribir consultas, escoger visualmente el motor de búsqueda y recibir los resultados enriquecidos con puntuaciones matemáticas (score TF-IDF o similitud coseno) y palabras clave resaltadas.
3. **Múltiples motores de resolución:** La herramienta integra metodologías distintas para encontrar la información, adaptándose a aproximaciones léxicas, semánticas y generativas.

Se ha elegido el modelo `es_core_news_md` frente a `es_core_news_sm` porque incluye embeddings preentrenados, algo de lo que carece la versión reducida (`sm`). Esto es fundamental para la **Búsqueda semántica**, ya que permite una representación y similitud de embeddings mucho más precisa en las consultas y fragmentos de texto, obteniendo mejores resultados de búsqueda.

### Modos de operación

1. **Búsqueda clásica**: se realiza búsqueda basada en similitud léxica, ignorando palabras "vacías" y se ordenan los resultados por relevancia utilizando el modelo TF-IDF.
2. **Búsqueda semántica**: se basa en embeddings vectoriales. Utiliza un modelo de lenguaje (spaCy) para capturar el significado contextual del texto y de la consulta del usuario, permitiendo encontrar resultados relevantes.
3. **RAG**: sistema de RAG (Retrieval-Augmented Generation) que combina los resultados de los modos de búsqueda clásica y semántica para generar respuestas en lenguaje natural usando un modelo de lenguaje (LLM) de Ollama.

## Configuración

### Variables de entorno
En primer lugar, puede ser necesario modificar el archivo ``.env`` para ajustar los valores de las variables de entorno. Por defecto se usan los siguientes:
```
# Configuración de spaCy
SPACY_MODEL=es_core_news_md

# Configuración de Ollama
OLLAMA_MODEL=llama3.2:1b

# Archivo de cache para embeddings
CACHE_FILE=embeddings_cache.json
```

## Ejecución

Instalar el wheel y ejecutarlo con los siguientes comandos:
```
uv tool install <ruta al wheel>
uv run fdi-pln-2606-p4
```

### Ollama para el modo de operación RAG
La documentación e instalación completa de Ollama puede encontrarse [aquí](https://ollama.com/). Para instalarlo, se pueden seguir los siguientes pasos:
```
wget https://ollama.com/download/ollama-linux-amd64.tar.zst

tar -xvf ollama-linux-amd64.tar.zst
```
Para iniciar Ollama, se utiliza el siguiente comando:
```
./bin/ollama serve
```
Es necesario iniciar Ollama en una terminal para poder ejecutar el **Modo RAG**.
