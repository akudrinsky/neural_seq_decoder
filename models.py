import torch
from torch import nn
import math
from augs import GaussianSmoothing

from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config


class NeuroT5(nn.Module):
    def __init__(self):
        super().__init__()

        if True:
            config = T5Config.from_pretrained("google-t5/t5-small")
            config.num_layers = 2
            self.t5 = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small", config=config)
        else:
            self.t5 = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        self.linear = nn.Linear(256 * 2, 512)
        print(self.t5)

        # for param in self.t5.parameters():
        #     param.requires_grad = False
        # for param in self.t5.encoder.parameters():
        #     param.requires_grad = True

    def _prepr_neuro(self, neuro, neuro_mask):
        B, T, C = neuro.shape
        if T % 2 != 0:
            neuro = F.pad(neuro, (0, 0, 0, 1))
            neuro_mask = F.pad(neuro_mask, (0, 1), value=0)
        T = neuro.size(1)
        neuro = neuro.reshape(B, T // 2, C * 2)
        neuro_mask = neuro_mask.view(B, T // 2, 2)
        neuro_mask = neuro_mask.max(dim=2).values

        return neuro, neuro_mask

    def forward(self, neuro, neuro_mask, labels=None):
        neuro, neuro_mask = self._prepr_neuro(neuro, neuro_mask)
        input_embeds = self.linear(neuro)

        outputs = self.t5(inputs_embeds=input_embeds, attention_mask=neuro_mask, labels=labels)
        return outputs


class GRUEncoder(nn.Module):
    def __init__(
        self,
        neural_dim=256,
        hidden_dim=1024,
        layer_dim=5,
        dropout=0.1,
        strideLen=4,
        kernelLen=32,
        gaussianSmoothWidth=2.0,
        bidirectional=True,
        output_embed_dim=30,
    ):
        super(GRUEncoder, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim

        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        if self.bidirectional:
            self.fc_out = nn.Linear(
                hidden_dim * 2, output_embed_dim
            )
        else:
            self.fc_out = nn.Linear(hidden_dim, output_embed_dim)

    def forward(self, neuralInput):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(neuralInput, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                neuralInput.size(0),
                self.hidden_dim,
                device=neuralInput.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                neuralInput.size(0),
                self.hidden_dim,
                device=neuralInput.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())
        out = self.fc_out(hid)

        return out


class ConvSequenceModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvSequenceModel, self).__init__()
        
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            # Convolution layer 1
            nn.Conv1d(input_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            # Convolution layer 2
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),

            # Convolution layer 3
            nn.Conv1d(512, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_channels),
            nn.Dropout(0.1),
        )
        
        # Adaptive pooling to reduce to 1 in the sequence dimension
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Permute x to (B, C, T) for 1D convolution
        x = x.permute(0, 2, 1)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Apply adaptive pooling (resulting shape: (B, C_out, 1))
        x = self.adaptive_pool(x)
        
        # Remove the last dimension (B, C_out)
        x = x.squeeze(-1)
        
        return x


class TransformerSequenceEmbedding(nn.Module):
    def __init__(self, input_dim=256, embed_dim=768, num_heads=12, num_layers=4, dropout_rate=0.1):
        super(TransformerSequenceEmbedding, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Layer to expand and project the input to the embedding dimension
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(embed_dim, dropout_rate)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer to convert the sequence of embeddings to a single embedding per batch
        self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x, mask):
        # Apply the linear projection
        x = self.embedding(x)
        
        # Add positional encodings
        x = self.positional_encoding(x)
        
        # Pass through the transformer
        x = self.transformer_encoder(x, src_key_padding_mask=~mask)
        
        # Pool the output to a fixed size embedding
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pooling(x).squeeze(2)  # (B, C)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


