---
title: Arquitetura — LLM Med QA
type: doc
created: 2026-06-07
---

# Arquitetura — LLM Med QA

## Visão Geral

```
MedAlpaca Dataset (HuggingFace)
         ↓
   Data Pipeline (src/data/)
         ↓
   QLoRA Fine-tuning (Kaggle T4×2)
   ┌─────────────────────────────┐
   │  Llama 3.2 3B Instruct      │
   │  + BitsAndBytes 4-bit NF4   │
   │  + LoRA r=16 α=32           │
   └─────────────────────────────┘
         ↓
   LoRA Adapter (HuggingFace Hub)
   LucasSGarrido/llama-3.2-3b-medqa
         ↓
   Gradio Interface (HF Spaces)
   app/app.py
```

## Componentes

| Módulo | Localização | Responsabilidade |
|--------|-------------|-----------------|
| Dataset | `src/data/dataset.py` | Carrega MedAlpaca, formata para Llama chat template |
| Config | `src/training/config.py` | Dataclasses de LoRA, treino e quantização |
| Métricas | `src/evaluation/metrics.py` | ROUGE-L e Exact Match |
| Inferência | `src/inference/pipeline.py` | Pipeline de Q&A com modelo fine-tuned |
| Notebook | `notebooks/finetune_qlora_kaggle.ipynb` | Treino completo no Kaggle T4×2 |
| App | `app/app.py` | Interface Gradio para HuggingFace Spaces |

## Decisões Técnicas

### QLoRA vs LoRA Full
QLoRA (4-bit NF4 + LoRA) foi escolhido porque Llama 3.2 3B em full precision requer ~12 GB VRAM. Com 4-bit quantização via bitsandbytes, o footprint cai para ~4 GB, cabendo confortavelmente em T4 (16 GB). O custo: velocidade ~20% menor que LoRA em fp16.

### Por que apenas q/v/k/o_proj?
Target modules limitados às projeções de atenção (não MLP) porque o dataset médico é relativamente pequeno (8k exemplos). Menos parâmetros treináveis reduz risco de overfitting e acelera treino.

### SFTTrainer vs Trainer vanilla
SFTTrainer (TRL) gerencia automaticamente packing de sequências, evitando padding excessivo. Com `dataset_text_field='text'`, a formatação do chat template já está embutida nas strings.

### Importações lazy com try/except no módulo
`src/inference/pipeline.py` usa try/except no nível do módulo (não dentro de funções) para que `patch()` do pytest possa substituir `AutoTokenizer` e `AutoModelForCausalLM` em CI sem GPU — mantendo os testes rápidos e isolados.

## Fluxo de Dados

```
instruction + input + output (MedAlpaca)
           ↓
   format_instruction()
           ↓
<|begin_of_text|>...<|eot_id|>  (Llama 3.2 chat template)
           ↓
   SFTTrainer (max_seq_length=1024)
           ↓
   LoRA adapter weights
           ↓
   HuggingFace Hub push
```
