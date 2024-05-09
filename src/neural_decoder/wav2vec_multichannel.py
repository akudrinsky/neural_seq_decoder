import torch
import transformers
from transformers.models.wav2vec2.modeling_wav2vec2 import *
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices


class Wav2Vec2ConfigMultichannel(Wav2Vec2Config):
    def __init__(self, num_signal_channels=1, **kwargs):
        super().__init__(**kwargs)
        self.num_signal_channels = num_signal_channels
transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Config = Wav2Vec2ConfigMultichannel


class Wav2Vec2NoLayerNormConvLayerMultichannel(Wav2Vec2NoLayerNormConvLayer):
    def __init__(self, config, layer_id=0):
        super().__init__(config, layer_id)
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else config.num_signal_channels
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]
transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer = Wav2Vec2NoLayerNormConvLayerMultichannel


class Wav2Vec2NoLayerNormConvLayerMultichannel(Wav2Vec2GroupNormConvLayer):
    def __init__(self, config, layer_id=0):
        super().__init__(config, layer_id)
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else config.num_signal_channels
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)


class Wav2Vec2GroupNormConvLayerMultichannel(Wav2Vec2GroupNormConvLayer):
    def __init__(self, config, layer_id=0):
        super().__init__(config, layer_id)
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else config.num_signal_channels
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)


class Wav2Vec2FeatureEncoderMultichannel(Wav2Vec2FeatureEncoder):
    """Construct the features from raw audio waveform"""

    def __init__(self, config):
        super().__init__(config)

        if config.feat_extract_norm == "group":
            conv_layers = [Wav2Vec2GroupNormConvLayerMultichannel(config, layer_id=0)] + [
                Wav2Vec2GroupNormConvLayerMultichannel(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states
transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder = Wav2Vec2FeatureEncoderMultichannel


def compute_mask_indices(**kwargs):
    return _compute_mask_indices(**kwargs)

def sample_negative_indices(**kwargs):
    return _sample_negative_indices(**kwargs)
