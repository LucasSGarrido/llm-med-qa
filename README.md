# LLM Med QA — Fine-tuning Llama 3.2 3B com QLoRA

> Fine-tuning eficiente de LLM para resposta a questões médicas. Do dataset ao deploy: QLoRA no Kaggle, adapter no HuggingFace Hub, interface Gradio ao vivo.

[![CI](https://github.com/LucasSGarrido/llm-med-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/LucasSGarrido/llm-med-qa/actions)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-llama--3.2--3b--medqa-yellow)](https://huggingface.co/SoulLucas/llama-3.2-3b-medqa)

---

## O Problema

Modelos de linguagem gerais cometem erros em questões médicas especializadas — confundem dosagens, mecanismos farmacológicos e diagnósticos diferenciais. Fine-tuning dirigido em um corpus médico curado pode reduzir esses erros de forma mensurável.

**Este projeto demonstra o pipeline completo de fine-tuning eficiente de LLM:** QLoRA para caber em GPU free tier, SFTTrainer para simplificar o loop de treino, e deploy público via HuggingFace Spaces.

---

## Demo ao Vivo

**[Abrir no HuggingFace Spaces](https://huggingface.co/spaces/SoulLucas/llm-med-qa)** — disponível após treino

---

## Arquitetura

```
MedAlpaca (HuggingFace)
       ↓
  format_instruction()        ← Llama 3.2 chat template
       ↓
  QLoRA Fine-tuning            ← Kaggle T4×2, 4-bit NF4, LoRA r=16
       ↓
  SoulLucas/llama-3.2-3b-medqa  ← HuggingFace Hub
       ↓
  Gradio Interface              ← HuggingFace Spaces
```

---

## Stack

| Categoria | Tecnologia |
|-----------|-----------|
| Modelo base | Llama 3.2 3B Instruct (Meta) |
| Fine-tuning | QLoRA · PEFT · TRL SFTTrainer |
| Quantização | bitsandbytes 4-bit NF4 |
| Dataset | MedAlpaca / medical_meadow_medqa |
| Interface | Gradio |
| Deploy | HuggingFace Spaces + Hub |
| Treino | Kaggle T4×2 (free tier) |
| CI | GitHub Actions |
| Qualidade | pytest · ruff |

---

## Dataset

**MedAlpaca — Medical Meadow MedQA**
- Fonte: [`medalpaca/medical_meadow_medqa`](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa)
- Questões estilo USMLE (United States Medical Licensing Examination)
- 8.000 exemplos de treino · 500 de validação

---

## Métricas

> Validação (n=100) — após treino no Kaggle

| Métrica | Valor |
|---------|-------|
| **ROUGE-L** | A preencher |
| **Loss de validação** | A preencher |

---

## Como Executar Localmente

### Pré-requisitos
- Python 3.10+
- GPU com >= 12 GB VRAM (para inferência com modelo completo)

### 1. Clonar e instalar

```bash
git clone https://github.com/LucasSGarrido/llm-med-qa.git
cd llm-med-qa
pip install -r requirements.txt
```

### 2. Rodar testes (sem GPU)

```bash
pytest tests/ -q
```

### 3. Interface Gradio

```bash
python app/app.py
```

### 4. Treino (Kaggle)

Fazer upload de `notebooks/finetune_qlora_kaggle.ipynb` no Kaggle, ativar T4×2, adicionar `HF_TOKEN` como Kaggle Secret e executar.

---

## Estrutura do Projeto

```
llm-med-qa/
├── src/
│   ├── data/          # Pipeline de dataset (MedAlpaca + chat template)
│   ├── training/      # Configs LoRA e treino (dataclasses)
│   ├── evaluation/    # ROUGE-L e Exact Match
│   └── inference/     # Pipeline de inferência
├── tests/             # pytest TDD (18 testes, sem GPU)
├── notebooks/         # Notebook Kaggle T4×2
├── app/               # Gradio para HuggingFace Spaces
├── config/            # training_config.yaml (QLoRA hyperparams)
├── docs/              # Arquitetura, model card
└── .github/workflows/ # CI GitHub Actions
```

---

## Documentação

- [Arquitetura detalhada](docs/arquitetura.md)
- [Model Card](docs/model_card.md)
- [Notebook Kaggle](notebooks/finetune_qlora_kaggle.ipynb)

---

## Aviso

Este modelo é para fins **educacionais e de portfólio**. Não usar como substituto de diagnóstico ou aconselhamento médico profissional.

---

## Autor

**Lucas Santos Garrido** — [LinkedIn](https://linkedin.com/in/lucas-garrido-8a119236a) · [GitHub](https://github.com/LucasSGarrido)
