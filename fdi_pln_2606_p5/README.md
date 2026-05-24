# P5 — Transformer LLM + NER (Lewis Carroll)

Transformer causal pre-entrenado sobre el corpus de Lewis Carroll (*Alice in Wonderland* + *Through the Looking-Glass*) con una cabeza NER para detectar personas y lugares.

## Integrantes

| Nombre | Usuario |
|--------|---------|
|        |         |

---

## Requisitos

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) — gestor de entornos y dependencias

## Instalación

```bash
uv sync
```

Esto crea el entorno virtual `.venv/` e instala todas las dependencias declaradas en `pyproject.toml` (torch, click, loguru, rich).

---

## Reproducción completa desde cero

Los pesos (`.pth`) **no están en el repositorio**. Para reproducir el experimento completo:

### 1. Entrenar el LLM causal

```bash
uv run fdi-pln-2606-p5 train-llm --corpus resources --weights-dir model_info
```

Parámetros principales (con sus valores por defecto):

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--corpus` | `resources` | Directorio con los ficheros `.txt` del corpus |
| `--weights-dir` | `model_info` | Directorio donde se guardan pesos, tokenizador y config |
| `--epochs` | `4` | Épocas de entrenamiento |
| `--lr` | `3e-4` | Tasa de aprendizaje |
| `--batch-size` | `40` | Tamaño de batch |
| `--d-model` | `128` | Dimensión del modelo |
| `--n-layers` | `3` | Número de bloques Transformer |
| `--n-heads` | `4` | Cabezas de atención |
| `--vocab-size` | `300` | Tamaño del vocabulario BPE |
| `--seq-len` | `128` | Longitud máxima de secuencia |

Genera en `--weights-dir`: `model.pth`, `tokenizer.json`, `config.json`.

### 2. Fine-tune del NER

```bash
uv run fdi-pln-2606-p5 train-ner --data merged_2.json --weights-dir model_info
```

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--data` | `merged_2.json` | Fichero JSON con el corpus etiquetado en BIO |
| `--weights-dir` | `model_info` | Mismo directorio que el LLM (carga backbone) |
| `--epochs` | `20` | Épocas de fine-tuning |
| `--lr` | `1e-3` | Tasa de aprendizaje (mejor config. según hypersearch) |
| `--batch-size` | `4` | Tamaño de batch |
| `--freeze-backbone` | `False` | Congelar pesos del backbone durante el fine-tuning |

Genera en `--weights-dir`: `ner_model.pth`.

---

## Uso con pesos ya entrenados

Si se dispone de los ficheros de pesos, colocarlos en un directorio (p. ej. `model_info/`) con esta estructura:

```
model_info/
  config.json          # arquitectura e hiperparámetros del backbone
  tokenizer.json       # vocabulario BPE (vocab_size=300)
  model.pth            # pesos del LLM causal  (o p5_causal_2606.pth)
  ner_model.pth        # pesos del modelo NER
```

### Generar texto

```bash
uv run fdi-pln-2606-p5 generate "alice looked at the"
uv run fdi-pln-2606-p5 generate "the queen said" --temperature 0.4 --max-tokens 100
```

Pasar el `.pth` directamente (sin depender de la estructura de directorio):

```bash
uv run fdi-pln-2606-p5 generate "alice" \
    --weights-dir model_info \
    --llm-path /ruta/al/model.pth
```

### Extraer entidades nombradas

```bash
uv run fdi-pln-2606-p5 entities texto.txt
```

Pasar el `.pth` directamente:

```bash
uv run fdi-pln-2606-p5 entities texto.txt \
    --weights-dir model_info \
    --ner-path /ruta/al/ner_model.pth
```

---

## Exploración de hiperparámetros

```bash
uv run python ner_hypersearch.py
```

Grid search de 12 configuraciones (`lr` × `freeze_backbone` × `batch_size`) con análisis previo del corpus. Resultados en `logs/hypersearch_results.json`.

---

## Estructura del proyecto

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
| `utils.py` | `load_corpus`, utilidades de ficheros |
| `main.py` | CLI unificado (4 comandos) |
| `fdi_pln_2606_p5/cli.py` | Entry point del paquete instalable |
| `merged_2.json` | Corpus etiquetado en esquema BIO (dato de entrenamiento) |
| `resources/` | Corpus de texto plano (Carroll) |
| `informe_2606.html` | Informe de exploración de hiperparámetros |
