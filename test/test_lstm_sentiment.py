import pytest
import torch.nn as nn
from model import LSTMSentiment

def test_load_save_state():
    model = LSTMSentiment(
        0, 2,
        300, 150,
        3, 'lstm', nn.NLLLoss()
    )
    model2 = LSTMSentiment(
        0, 2,
        300, 150,
        3, 'lstm', nn.NLLLoss()
    )
    model.save_state_files('.')
    model2.load_state_files('.')
    model3 = LSTMSentiment(
        0, 1,
        300, 150,
        3, 'lstm', nn.NLLLoss()
    )

    with pytest.raises(Exception):
        model3.load_state_files('.')

