---
title: Model Card — llama-3.2-3b-medqa
type: doc
created: 2026-06-07
---

# Model Card — SoulLucas/llama-3.2-3b-medqa

## Descrição do Modelo

LoRA adapter sobre **Llama 3.2 3B Instruct** (Meta), fine-tuned com QLoRA no dataset MedAlpaca para resposta a questões médicas estilo USMLE.

## Informações de Treino

| Parâmetro | Valor |
|-----------|-------|
| Modelo base | meta-llama/Llama-3.2-3B-Instruct |
| Dataset | medalpaca/medical_meadow_medqa |
| Amostras de treino | 8.000 |
| Amostras de validação | 500 |
| Épocas | 2 |
| Batch size | 4 (grad acc: 4 → efetivo: 16) |
| Learning rate | 2e-4 |
| Hardware | Kaggle T4×2 (16 GB VRAM cada) |
| Quantização | 4-bit NF4 (bitsandbytes) |
| LoRA rank | r=16, α=32 |
| Target modules | q_proj, v_proj, k_proj, o_proj |

## Avaliação

| Métrica | Conjunto | Valor |
|---------|----------|-------|
| ROUGE-L | Validação (n=100) | A preencher após treino |
| Loss de validação | Val set (500 ex.) | A preencher após treino |

## Uso

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("SoulLucas/llama-3.2-3b-medqa")
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "SoulLucas/llama-3.2-3b-medqa")
model.eval()
```

## Limitações e Avisos

- **Uso educacional apenas.** Não usar como substituto de diagnóstico ou aconselhamento médico profissional.
- O modelo foi treinado em questões USMLE estilo multiple-choice; pode ter desempenho variável em perguntas abertas muito específicas.
- Alucinações são possíveis — sempre verificar com fontes médicas primárias.

## Dataset

**MedAlpaca / Medical Meadow MedQA**
- Fonte: `medalpaca/medical_meadow_medqa` (HuggingFace)
- Conteúdo: questões de exame médico USMLE
- Licença: verificar no repositório original
