# The MIT License (MIT)
#
# Copyright (c) 2021 Playtika Ltd.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# *** INSTRUCTIONS ***
# For the test, the following packages need to be installed:
# pip install pytest
# pip install pytest-cov
# ---------------
# Also, use torch version 1.7.1: some functions do not work with torch version 1.9.0
# ---------------
# To launch this test file only, run the following command:
# pytest tests/utils/test_embedding.py
# To launch all tests inside /esntorch/utils/ with line coverage, run the following command:
# pytest --cov tests/utils/
# *** END INSTRUCTIONS ***

from esntorch.utils.embedding import EmbeddingModel
import torch
import transformers
# import pytest


def test_init():
    """Tests the __init__ method."""

    embedding = EmbeddingModel(model_name='bert-base-uncased', device=torch.device('cpu'))

    assert embedding.model_name == 'bert-base-uncased'
    assert isinstance(embedding.device, torch.device)
    assert isinstance(embedding.model, transformers.models.bert.modeling_bert.BertModel)
    assert not embedding.model.training


def test_get_embedding():
    """Tests the get_embedding method."""

    device = torch.device('cpu')
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    batch = tokenizer(["This is a first sentence.",
                       "And here is a second one.",
                       "A third sentence..."],
                      padding=True,
                      return_tensors="pt")

    embedding = EmbeddingModel(model_name='bert-base-uncased', device=device)

    batch = batch.to(device)
    batch_emb = embedding.model(batch["input_ids"], batch["attention_mask"])[0].transpose(0, 1)

    assert batch_emb.shape[1] == 3    # nb of sentences
    assert batch_emb.shape[2] == 768  # embedding dim
