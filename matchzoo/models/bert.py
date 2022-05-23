"""An implementation of Bert Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from matchzoo import preprocessors
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_preprocessor import BasePreprocessor
from matchzoo.engine import hyper_spaces
from matchzoo.dataloader import callbacks
from matchzoo.modules import TransformersModule, Pooling


class Bert(BaseModel):
    """Bert Model."""

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params.add(Param(name='mode', value='bert-base-uncased',
                         desc="Pretrained Bert model."))
        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        params.add(Param(
            name="out_type", value="score", desc="output of bert model, can be left/right/score"
        ))
        params.add(Param(
            name="pooling_mode", value="mean", desc="pooing mode, can be a string: mean/max/cls"
        ))
        return params

    @classmethod
    def get_default_preprocessor(
        cls,
        mode: str = 'bert-base-uncased'
    ) -> BasePreprocessor:
        """:return: Default preprocessor."""
        return preprocessors.BertPreprocessor(mode=mode)

    @classmethod
    def get_default_padding_callback(
        cls,
        fixed_length_left: int = None,
        fixed_length_right: int = None,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre'
    ):
        """:return: Default padding callback."""
        return callbacks.BertPadding(
            fixed_length_left=fixed_length_left,
            fixed_length_right=fixed_length_right,
            pad_value=pad_value,
            pad_mode=pad_mode)

    def build(self):
        """Build model structure."""
        self.bert = TransformersModule(mode=self._params['mode'])
        self.dropout = nn.Dropout(p=self._params['dropout_rate'])
        self.pooling = Pooling(word_embedding_dimension=self.bert.bert.embeddings.word_embeddings.num_embeddings,
                               pooling_mode=self._params["pooling_mode"])

    def get_vector(self, x):
        input_ids = x
        token_type_ids = torch.zeros_like(x)
        attention_mask = (input_ids != 0)
        vector = self.bert(input_ids, token_type_ids, attention_mask)
        features = {"token_embeddings": vector[0], "cls_token_embeddings": vector[1], "attention_mask": attention_mask}
        embedding_result = self.pooling(features)
        return embedding_result["sentence_embedding"]

    def forward(self, inputs, do_eval=False):
        """Forward."""

        input_left, input_right = inputs['text_left'], inputs['text_right']
        left_vector = self.get_vector(input_left)
        right_vector = self.get_vector(input_right)

        if do_eval:
            return F.cosine_similarity(left_vector, right_vector)
        else:
            if self._params["out_type"] == "left":
                out = left_vector
            elif self._params["out_type"] == "right":
                out = right_vector
            elif self._params["out_type"] == "score":
                return F.cosine_similarity(left_vector, right_vector)
            elif self._params["out_type"] == "supervised":
                batch_size, hidden_dim = left_vector.shape
                batch_vector = torch.cat((left_vector, right_vector), dim=1).reshape((batch_size*2, hidden_dim))
                # [x, x+, x, x-] ---> [x, x+, x-]
                used_index = torch.where(torch.arange(batch_size*2, device=left_vector.device) % 4 != 2)[0]
                out = batch_vector.index_select(dim=0, index=used_index)  # shape=(3*batch_size, hidden_dim)
            elif self._params["out_type"] == "unsupervised":
                out = left_vector   # �޼ල left==right
            else:
                raise NotImplementedError

            out = self.dropout(out)
            return out
