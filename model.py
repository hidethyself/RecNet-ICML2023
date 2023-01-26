import torch
import torch.nn as nn
from seldnet_model import CRNN
from early_attention.attention import EarlyAttention
from interpolator.deep_interpolator import DeepInterpolator


class ModelConfig:
    def __init__(
            self,
            type_,
            input_shape,
            out_shape,
            parameters
    ):
        self.type_ = type_
        self.input_shape = input_shape
        self.output_shape = out_shape
        self.params = parameters

    def load_model(self):
        if self.type_ == 1:
            model = CRNN(in_feat_shape=self.input_shape, out_shape=self.output_shape, params=self.params)
            return model
        # elif self.type_ == 2:
        #     model = EarlyAttention(data_in=self.input_shape, data_out=self.output_shape, params=self.params)
        #     return model
        elif self.type_ == 2:
            model = EarlyAttention(data_in=self.input_shape, data_out=self.output_shape, params=self.params)
            interpolator = DeepInterpolator()
            return model, interpolator


