"""
Gradio interface for the LLM Med QA fine-tuned model.
Deployed on HuggingFace Spaces.
"""
from __future__ import annotations

import os

import gradio as gr

# Lazy imports — model loading happens only when the Space starts
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

MODEL_ID = os.getenv("BASE_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
ADAPTER_ID = os.getenv("ADAPTER_ID", "SoulLucas/llama-3.2-3b-medqa")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. "
    "Answer questions accurately based on medical knowledge."
)

DESCRIPTION = """
# 🩺 Med QA — Llama 3.2 3B Fine-tuned

Modelo Llama 3.2 3B Instruct ajustado com QLoRA no dataset **MedAlpaca** (questões estilo USMLE).

**Dataset:** `medalpaca/medical_meadow_medqa` · **Treino:** 8.000 exemplos · **LoRA:** r=16, α=32
**⚠️ Aviso:** Este modelo é para fins educacionais. Não usar como substituto de aconselhamento médico profissional.
"""

EXAMPLES = [
    ["What is the mechanism of action of aspirin?"],
    ["What are the first-line treatments for type 2 diabetes?"],
    ["Explain the difference between bacterial and viral infections."],
    ["What is the pathophysiology of myocardial infarction?"],
    ["What are the symptoms of pulmonary embolism?"],
]

# Module-level model/tokenizer (loaded once at startup)
_tokenizer = None
_model = None
_load_error = None


def _load_model() -> str | None:
    """Load model and tokenizer. Returns error message or None on success."""
    global _tokenizer, _model, _load_error

    if not HAS_DEPS:
        _load_error = "Missing dependencies: torch, transformers, or peft not installed."
        return _load_error

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
        _tokenizer.pad_token = _tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        _model = PeftModel.from_pretrained(base, ADAPTER_ID)
        _model.eval()
        return None  # success
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)
        return _load_error


def _build_prompt(question: str, context: str = "") -> str:
    user_content = question
    if context.strip():
        user_content = f"Context: {context}\n\nQuestion: {question}"
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def answer(question: str, context: str = "") -> str:
    """Generate answer for a medical question."""
    if _load_error:
        return f"❌ Erro ao carregar o modelo: {_load_error}"
    if _model is None or _tokenizer is None:
        return "⏳ Modelo ainda carregando..."

    question = question.strip()
    if not question:
        return "Por favor, insira uma pergunta."

    prompt = _build_prompt(question, context)
    inputs = _tokenizer(prompt, return_tensors="pt")
    if hasattr(_model, "device"):
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Med QA — Llama 3.2 3B") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Pergunta Médica",
                    placeholder="Ex: What is the mechanism of action of aspirin?",
                    lines=3,
                )
                context_input = gr.Textbox(
                    label="Contexto (opcional)",
                    placeholder="Ex: Patient presents with chest pain...",
                    lines=2,
                )
                submit_btn = gr.Button("Responder", variant="primary")

            with gr.Column(scale=2):
                answer_output = gr.Textbox(
                    label="Resposta",
                    lines=8,
                    interactive=False,
                )

        gr.Examples(
            examples=EXAMPLES,
            inputs=question_input,
            label="Exemplos",
        )

        submit_btn.click(
            fn=answer,
            inputs=[question_input, context_input],
            outputs=answer_output,
        )
        question_input.submit(
            fn=answer,
            inputs=[question_input, context_input],
            outputs=answer_output,
        )

    return demo


if __name__ == "__main__":
    _load_model()
    demo = build_interface()
    demo.launch()
