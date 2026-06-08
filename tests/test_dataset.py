from unittest.mock import MagicMock, patch

from src.data.dataset import SYSTEM_PROMPT, format_instruction


def test_format_instruction_without_input(medalpaca_example):
    """format_instruction sem input: user content é só a instruction."""
    result = format_instruction(medalpaca_example)
    text = result["text"]
    assert SYSTEM_PROMPT in text
    assert medalpaca_example["instruction"] in text
    assert medalpaca_example["output"] in text
    assert "<|begin_of_text|>" in text
    assert "<|eot_id|>" in text


def test_format_instruction_with_input(medalpaca_example_with_input):
    """format_instruction com input: instruction e input aparecem no user turn."""
    result = format_instruction(medalpaca_example_with_input)
    text = result["text"]
    assert medalpaca_example_with_input["instruction"] in text
    assert medalpaca_example_with_input["input"] in text
    assert medalpaca_example_with_input["output"] in text


def test_format_instruction_returns_only_text_key(medalpaca_example):
    """format_instruction retorna dict com SOMENTE a chave 'text'."""
    result = format_instruction(medalpaca_example)
    assert set(result.keys()) == {"text"}


def test_load_medalpaca_calls_hf():
    """load_medalpaca chama load_dataset com DATASET_ID correto."""
    from src.data.dataset import DATASET_ID, load_medalpaca

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: 100
    mock_ds.shuffle.return_value.select.return_value = mock_ds
    with patch("src.data.dataset.load_dataset", return_value=mock_ds) as mock_load:
        load_medalpaca(max_samples=10)
        mock_load.assert_called_once_with(DATASET_ID, split="train")
