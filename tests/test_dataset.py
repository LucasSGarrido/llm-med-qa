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


def test_prepare_dataset_returns_train_val_split():
    """prepare_dataset retorna DatasetDict com chaves 'train' e 'validation'."""
    from datasets import Dataset, DatasetDict

    from src.data.dataset import prepare_dataset

    # Create a small mock dataset with 20 examples
    mock_data = [
        {"instruction": f"Question {i}?", "input": "", "output": f"Answer {i}."}
        for i in range(20)
    ]
    mock_ds = Dataset.from_list(mock_data)

    # prepare_dataset calls load_medalpaca which calls load_dataset
    # We mock at the load_dataset level
    with patch("src.data.dataset.load_dataset", return_value=mock_ds):
        result = prepare_dataset(max_train=15, max_val=5, seed=42)

    assert isinstance(result, DatasetDict)
    assert set(result.keys()) == {"train", "validation"}
    # Combined size should be around max_train + max_val (may vary slightly due to shuffle)
    assert len(result["train"]) + len(result["validation"]) == 20
    # Each example should have only the "text" key after formatting
    assert list(result["train"].column_names) == ["text"]
    assert list(result["validation"].column_names) == ["text"]
