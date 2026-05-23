# P5 — Transformer LLM + NER (Lewis Carroll)

Transformer causal pre-entrenado sobre el corpus de Lewis Carroll con una cabeza NER para detectar personas y lugares.

## Requisitos

```bash
uv sync          # instala dependencias de pyproject.toml
```

## Pesos pre-entrenados del backbone

Los ficheros `.pth` no están en el repositorio. Coloca los siguientes ficheros en `model_info/` antes de entrenar el NER:

```
model_info/
  config.json          # arquitectura del backbone
  tokenizer.json       # vocabulario BPE
  p5_causal_2606.pth   # pesos del backbone (LLM causal)
```

## Entrenar el NER

```bash
uv run python main.py train-ner --data merged_2.json --weights-dir model_info
```

Parámetros configurables (`--lr`, `--epochs`, `--batch-size`):

```bash
uv run python main.py train-ner --data merged_2.json --weights-dir model_info --lr 5e-4 --epochs 15 --batch-size 8
uv run python main.py train-ner --data merged_2.json --weights-dir model_info --lr 1e-4 --epochs 30 --batch-size 8
uv run python main.py train-ner --data merged_2.json --weights-dir model_info --lr 5e-5 --epochs 15 --batch-size 4
```

La pérdida por época se imprime en consola y se guarda en `logs/ner_loss.txt`.

## Entrenar el backbone LLM desde cero

```bash
uv run python main.py train-llm --corpus resources --weights-dir model_info
```

## Inferencia

```bash
# Generar texto
uv run python main.py generate "alice was" --weights-dir model_info

# Extraer entidades de un fichero
uv run python main.py entities resources/alice_in_wonderland.txt --weights-dir model_info
```

## Estructura

| Fichero | Descripción |
|---|---|
| `transformer.py` | Bloque Transformer base (atención + FFN) |
| `causal_llm.py` | Modelo LLM causal (máscara triangular) |
| `causal_train.py` | Bucle de pre-entrenamiento LLM |
| `ner.py` | Modelo NERLLM, dataset y alineamiento BPE |
| `ner_train.py` | Fine-tuning NER (`train_ner`) |
| `main.py` | CLI unificado (train-llm / train-ner / generate / entities) |
| `tokenizer.py` | Tokenizador BPE |
| `utils.py` | `load_corpus`, `save_losses` |
