from __future__ import annotations

# Lazy module-level stubs — allow patching in tests without requiring GPU deps.
# The real imports are attempted here; if unavailable (CI without torch/transformers),
# the names are set to None and replaced by patch() in tests.
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment,misc]
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]

SYSTEM_PROMPT = (
    "You are a helpful medical assistant. "
    "Answer questions accurately based on medical knowledge."
)


class MedQAPipeline:
    """Inference pipeline for the fine-tuned Llama 3.2 3B medical Q&A model.

    Heavy imports (torch, transformers) are attempted at module level but
    gracefully degrade to None in CI environments without GPU dependencies.
    Tests patch the module-level names directly.
    """

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        self.model.eval()

    @staticmethod
    def _build_prompt(question: str, context: str = "") -> str:
        """Build Llama 3.2 Instruct chat prompt for a medical question."""
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

    def answer(self, question: str, context: str = "") -> str:
        """Generate an answer for the given medical question.

        Args:
            question: Medical question text.
            context: Optional clinical context (e.g., patient symptoms).

        Returns:
            Generated answer string, decoded and stripped.
        """
        if torch is None:  # pragma: no cover
            raise ImportError(
                "torch is required to run inference. Install requirements-training.txt."
            )
        prompt = self._build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
