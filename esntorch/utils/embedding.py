#   Copyright 2021 Playtika Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
