from unittest.mock import MagicMock, patch

from src.inference.pipeline import SYSTEM_PROMPT, MedQAPipeline


def test_build_prompt_without_context():
    """_build_prompt sem context: question aparece no user turn, sem 'Context:'."""
    prompt = MedQAPipeline._build_prompt("What is aspirin?")
    assert SYSTEM_PROMPT in prompt
    assert "What is aspirin?" in prompt
    assert "Context:" not in prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt


def test_build_prompt_with_context():
    """_build_prompt com context: context e question aparecem separados."""
    prompt = MedQAPipeline._build_prompt(
        "What is the diagnosis?", context="Patient has fever and cough."
    )
    assert "Context: Patient has fever and cough." in prompt
    assert "Question: What is the diagnosis?" in prompt


def test_pipeline_init_loads_model_and_tokenizer():
    """__init__ carrega tokenizer e model via lazy imports."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model

    with patch("src.inference.pipeline.AutoTokenizer") as mock_tok_cls, \
         patch("src.inference.pipeline.AutoModelForCausalLM") as mock_model_cls, \
         patch("src.inference.pipeline.torch"):
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        pipe = MedQAPipeline("fake/model-id")

        mock_tok_cls.from_pretrained.assert_called_once_with("fake/model-id")
        assert pipe.model_id == "fake/model-id"
        assert pipe.max_new_tokens == 256


def test_answer_returns_decoded_string():
    """answer() chama generate() e decode(), retorna string stripped."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.device = "cpu"
    mock_tokenizer.decode.return_value = "  Aspirin inhibits COX enzymes.  "

    with patch("src.inference.pipeline.AutoTokenizer") as mock_tok_cls, \
         patch("src.inference.pipeline.AutoModelForCausalLM") as mock_model_cls, \
         patch("src.inference.pipeline.torch") as mock_torch:
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_torch.no_grad.return_value.__enter__ = lambda s: s
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        pipe = MedQAPipeline("fake/model-id")
        result = pipe.answer("What is aspirin?")

        assert result == "Aspirin inhibits COX enzymes."
        mock_model.generate.assert_called_once()
