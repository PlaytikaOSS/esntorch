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
    A BERT object contain a BERT model and a method that implement the BERT embedding of batches.
    """

    def __init__(self, model_name='bert-base-uncased', device=torch.device('cpu')):
        """
        Constructor

        Parameters
        ----------
        model_name : str
            Name of the BERT model and tokenizer.

        Attributes
        ----------
        model_name : str
            Name of the BERT model and tokenizer.
            The list of or possible models is provided here: https://huggingface.co/models
        tokenizer : transformers.models.bert.tokenization_bert.BertTokenizer
            Tokenizer used in the word embedding process.
        model :
            Model used in the word embedding process.
        device :
            GPU is available, CPU otherwise.
        """

        self.device = device
        self.model_name = model_name
        self.model = BertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.model.to(self.device).eval()
        # print('BERT model downloaded:', model_name)

    def get_embedding(self, batch):
        """
        Embeds a batch of token ids into a 3D tensor.
        If a GPU is available, the embedded batch is computed and put on the GPU.

        Parameters
        ----------
        batch: torch.Tensor
            2D tensor: batch of text to be embedded.
            The sentences are represented by the vertical sequences of token ids.

        Returns
        -------
        batch_emb : torch.Tensor
            3D tensor (batch size x max sentence length x embedding dim)
            BERT embedding of the batch of texts.
        """

        with torch.no_grad():

            # method 1: dynamic embedding
            # batch_tkn = batch["input_ids"].to(self.device)
            batch = batch.to(self.device)
            batch_emb = self.model(batch["input_ids"], batch["attention_mask"])[0].transpose(0, 1)

            # method 2: concatenate last 2 layers
            # output = self.model(batch_tkn)
            # batch_emb_1 = output[2][-1]
            # batch_emb_2 = output[2][-2]
            # batch_emb = torch.cat([batch_emb_1, batch_emb_2], dim=2).transpose(0, 1)

            # method 3: static embedding (less powerful)
            # batch_emb = self.model.embeddings.word_embeddings(batch_tkn).transpose(0, 1)

        return batch_emb
