# tests/conftest.py
import pytest


@pytest.fixture
def medalpaca_example():
    return {
        "instruction": "What is the mechanism of action of aspirin?",
        "input": "",
        "output": (
            "Aspirin irreversibly inhibits cyclooxygenase (COX-1 and COX-2) enzymes, "
            "preventing the synthesis of prostaglandins and thromboxanes."
        ),
    }


@pytest.fixture
def medalpaca_example_with_input():
    return {
        "instruction": "Based on the following symptoms, what is the likely diagnosis?",
        "input": "Patient presents with chest pain, shortness of breath, and diaphoresis.",
        "output": (
            "The symptoms suggest acute myocardial infarction (heart attack). "
            "Immediate evaluation with ECG and troponin levels is warranted."
        ),
    }
