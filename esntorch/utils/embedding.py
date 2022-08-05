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

import torch
from transformers import BertModel


class EmbeddingModel():
    """
    An EmbeddingModel object contain a Hugging Face model
    and a method that embeds batches according to this model.

    Parameters
    ----------
    model_name : `str`
        Name of the Hugging Face model and tokenizer.
        The list of possible models is provided here: https://huggingface.co/models
    device : `torch.device`
        GPU if available, CPU otherwise.

    Attributes
    ----------
    device : `torch.device`
        GPU if available, CPU otherwise.
    model_name : `str`
        Name of the Hugging Face model.
        The list of or possible models is provided here: https://huggingface.co/models
    model : `transformers.PreTrainedModel`
        Hugging Face model used for embedding_fn.
    """

    def __init__(self, model_name='bert-base-uncased', device=torch.device('cpu')):
        self.device = device
        self.model_name = model_name
        self.model = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.model.to(self.device).eval()
        print('Model downloaded:', model_name)

    def get_embedding(self, batch):
        """
        Embeds a batch of token ids into a 3D tensor.
        If the device is a GPU, the embedded batch is put on GPU.

        Parameters
        ----------
        batch : `transformers.tokenization_utils_base.BatchEncoding`
            Batch of token ids.

        Returns
        -------
        batch_emb : `torch.Tensor`
            3D tensor [max sentence length x batch size x embedding_fn dim]
            Embedding of the batch of token ids.
        """

        with torch.no_grad():
            # method 1: dynamic embedding_fn - classical way
            # batch_tkn = batch["input_ids"].to(self.device)
            batch = batch.to(self.device)
            batch_emb = self.model(batch["input_ids"], batch["attention_mask"])[0].transpose(0, 1)

            # method 2: dynamic embedding_fn - concatenates the last two layers
            # output = self.model(batch_tkn)
            # batch_emb_1 = output[2][-1]
            # batch_emb_2 = output[2][-2]
            # batch_emb = torch.cat([batch_emb_1, batch_emb_2], dim=2).transpose(0, 1)

            # method 3: static embedding_fn (less powerful)
            # batch_emb = self.model.embeddings.word_embeddings(batch_tkn).transpose(0, 1)

        return batch_emb
