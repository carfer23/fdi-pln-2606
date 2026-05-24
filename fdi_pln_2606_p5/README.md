# P5 - Transformer LLM + NER

## Integrantes
- Carmen Fernández González
- Yushan Yang Xu

## Descripción

En esta práctica se implementa un Transformer causal pre-entrenado sobre el corpus de *Alice in Wonderland* + *Through the Looking-Glass* con un cabezal NER para detección de entidades, concretamente personas y lugares.

## Resumen de ficheros

| Fichero | Descripción |
|---------|-------------|
| `transformer.py` | Bloque Transformer base (atención multi-cabeza + FFN) |
| `attention.py` | Atención escalada con máscara causal opcional |
| `causal_llm.py` | Modelo LLM causal (máscara triangular, generación) |
| `causal_train.py` | Bucle de pre-entrenamiento |
| `ner.py` | Modelo NERLLM, dataset NER y alineamiento BPE↔palabra |
| `ner_train.py` | Fine-tuning NER con `class_weights` por desequilibrio |
| `ner_hypersearch.py` | Grid search de hiperparámetros NER |
| `tokenizer.py` | Tokenizador BPE entrenado sobre el corpus |
| `utils.py` | Funciones auxiliares |
| `defaults.py` | Hiperparámetros por defecto |
| `main.py` | CLI unificado (4 comandos) |

## Configuración por defecto

- Corpus (texto para tokenizador / LLM): carpeta `resources/`
- Etiquetas NER: `labels/ner_labels.json`
- Pesos LLM por defecto: `model_info/p5_causal_2606.pth`
- Salida tokenizador/config: `model_info/tokenizer.json`, `model_info/config.json`
- Pesos NER salida: `model_info/p5_ner_2606.pth`
- Logs: `logs/`

## Uso

El wheel se encuentra disponible en la *release* del repositorio. Instalar el wheel con el siguiente comando:
```
uv tool install <ruta al wheel>
```

El programa ofrece los siguientes comandos:
```
Commands:
  entities   Encuentra entidades nombradas (personas y lugares) en FILE.
  generate   Genera texto a partir de PROMPT usando el LLM entrenado.
  train-llm  Pre-entrena el LLM causal sobre el corpus de texto.
  train-ner  Fine-tune del cabezal NER sobre el corpus etiquetado.
```

Para obtener información detallada del uso de cada uno, ver el argumento `--help` de cada uno.

**Comandos principales y uso básico**
- train-llm — preentrena LLM y guarda artefactos:
  - Requiere: corpus de texto (`--corpus`, por defecto `resources/`).
  - Produce: `model_info/p5_causal_2606.pth`, `model_info/tokenizer.json`, `model_info/config.json`.
  - Ejemplo:
    ```bash
    fdi-pln-2606-p5 train-llm --corpus resources
    ```
- train-ner — fine-tuning de la cabeza NER sobre un backbone:
  - Requiere: fichero labels (`--labels`), fichero .pth del backbone (`--weights-path`, por defecto `model_info/p5_causal_2606.pth`), corpus para tokenizador (`--corpus`).
  - Opciones clave: `--freeze-backbone` (por defecto en main.py es False — pasar explícitamente si quieres congelarlo).  
  - Produce: `model_info/p5_ner_2606.pth`.
  - Ejemplo:
    ```bash
    fdi-pln-2606-p5 train-ner --labels labels/ner_labels.json --weights-path model_info/p5_causal_2606.pth --freeze-backbone
    ```
- generate — genera texto con el LLM:
  - Requiere: `.pth` del LLM (`--llm-path`) y corpus para entrenar tokenizador en tiempo de ejecución (`--corpus`) o `--config-path` para cargar hiperparámetros.  
  - Si no pasa `--config-path`, usa DEFAULTS (central).  
  - Ejemplo:
    ```bash
    fdi-pln-2606-p5 generate "alice opened the door" --llm-path model_info/p5_causal_2606.pth --corpus resources
    ```
- entities — extrae entidades usando modelo NER:
  - Requiere: `--ner-path` (por defecto `model_info/p5_ner_2606.pth`) y `--corpus` para construir tokenizer.  
  - Opciones: `--config-path` (opcional) y `--chunk-words`.  
  - Ejemplo:
    ```bash
    fdi-pln-2606-p5 entities path/to/text.txt --ner-path model_info/p5_ner_2606.pth --corpus resources
    ```
