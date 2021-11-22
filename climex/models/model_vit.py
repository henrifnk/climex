import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-headed Attention for input Query, Key, Value

    Multi-headed Attention is a module for attention mechanisms which runs through attention in several times in
    parallel, then the multiple outputs are concatenated and linearly transformed

    Attributes:
        embed_size (int): Embedding Size of Input
        num_heads (int): Number of heads in Multi-headed Attention; Number of Splits in the Embedding Size
        dropout (float, optional): Percentage of Dropout to be applied in range 0 <= dropout <=1
        batch_dim (int, optional): The dimension in which batch dimensions is

    """

    def __init__(self, embed_size, num_heads, dropout=0.2, batch_dim=0):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_dim = batch_dim

        self.dropout_layer = nn.Dropout(dropout)

        self.head_size = self.embed_size // self.num_heads

        assert self.head_size * self.num_heads == self.embed_size, "Heads cannot split Embedding size equally"

        self.Q = nn.Linear(self.embed_size, self.embed_size)
        self.K = nn.Linear(self.embed_size, self.embed_size)
        self.V = nn.Linear(self.embed_size, self.embed_size)

        self.linear = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, q, k, v, mask=None):
        if self.batch_dim == 0:
            out = self.batch_0(q, k, v, mask)
        elif self.batch_dim == 1:
            out = self.batch_1(q, k, v, mask)

        return out

    def batch_0(self, q, k, v, mask=None):
        q_batch_size, q_seq_len, q_embed_size = q.size()
        k_batch_size, k_seq_len, k_embed_size = k.size()
        v_batch_size, v_seq_len, v_embed_size = v.size()

        q = self.Q(q).reshape(q_batch_size, q_seq_len, self.num_heads, self.head_size)
        k = self.K(k).reshape(k_batch_size, k_seq_len, self.num_heads, self.head_size)
        v = self.V(v).reshape(v_batch_size, v_seq_len, self.num_heads, self.head_size)

        attention = self.attention(q, k, v, mask=mask)
        concatenated = attention.reshape(v_batch_size, -1, self.embed_size)
        out = self.linear(concatenated)

        return out

    def batch_1(self, q, k, v, mask=None):
        q_seq_len, q_batch_size, q_embed_size = q.size()
        k_seq_len, k_batch_size, k_embed_size = k.size()
        v_seq_len, v_batch_size, v_embed_size = v.size()

        q = self.Q(q).reshape(q_seq_len, q_batch_size, self.num_heads, self.head_size).transpose(0, 1)
        k = self.K(k).reshape(k_seq_len, k_batch_size, self.num_heads, self.head_size).transpose(0, 1)
        v = self.V(v).reshape(v_seq_len, v_batch_size, self.num_heads, self.head_size).transpose(0, 1)

        attention = self.attention(q, k, v, mask=mask)
        concatenated = attention.reshape(-1, v_batch_size, self.embed_size)

        out = self.linear(concatenated)

        return out

    def attention(self, q, k, v, mask=None):
        scores = torch.einsum("bqhe,bkhe->bhqk", [q, k])

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores /= math.sqrt(self.embed_size)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)
        attention = torch.einsum("bhql,blhd->bqhd", [scores, v])
        return attention


class VisionEncoder(nn.Module):
    """ Vision Encoder class for transformer

    Attributes:
        embed_size (int): Embedding Size of Input
        num_heads (int): Number of heads in Multi-headed Attention; Number of Splits in the Embedding Size
        hidden_size (int): size of hidden layers
        dropout (float, optional): Percentage of Dropout to be applied in range 0 <= dropout <=1
        norm1 (nn.LayerNorm): Hidden Layer
        norm2 (nn.LayerNorm): Hidden Layer
        attention (MultiHeadAttention): object, see class MultiHeadAttention
        mlp (nn.Sequential): A sequential container
    """
    def __init__(self, embed_size, num_heads, hidden_size, dropout=0.1):
        super(VisionEncoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.attention = MultiHeadAttention(self.embed_size, self.num_heads, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, 4 * self.embed_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.embed_size, self.embed_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attention(x, x, x)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer class

    Attributes:
        p (int): Patch size for ViT. Default is 4 dividing images in 4x4 patches
        image_height (int): Pixel height of the images.
            Default is 16.
        image_width (int): Pixel width of the images.
            Default is 39.
        embed_size (int): Embedding Size of Input, defaults to 512
        num_patches (float): Number of patches per image
        patch_size (int): The patch size, given both channels. Defaults to 32.
        classes (int): # labels
        num_layers (int): # of layers in network
        num_heads (int): Number of heads in Multi-headed Attention; Number of Splits in the Embedding Size
        hidden_size (int): size of hidden layers
        dropout (float, optional): Percentage of Dropout to be applied in range 0 <= dropout <=1
        dropout_layer (nn.Dropout): drop out rate
        embeddings (nn.Linear): Linear embedding
        class_token (nn.Parameter): random Parameter
        positional_encoding (nn.Parameter): random Parameter
        encoders (nn.ModuleList):VisionEncoder
        norm (nn.LayerNorm): Norm Layer
        classifier (nn.Sequential): Sequential classifier
    """
    def __init__(self, image_height=16, image_width=39, channel_size=2, patch_size=4,
                 embed_size=512, num_heads=8, classes=3, num_layers=2, hidden_size=265,
                 dropout=0.2):
        super(ViT, self).__init__()

        self.p = patch_size
        self.image_height = image_height
        self.image_width = image_width
        self.embed_size = embed_size
        self.num_patches = (image_height * image_width) / patch_size ** 2
        self.patch_size = channel_size * (patch_size ** 2)
        self.num_heads = num_heads
        self.classes = classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.embeddings = nn.Linear(self.patch_size, self.embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.positional_encoding = nn.Parameter(torch.randn(1, int(self.num_patches + 1), self.embed_size))

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(VisionEncoder(self.embed_size, self.num_heads, self.hidden_size, self.dropout))

        self.norm = nn.LayerNorm(self.embed_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.classes)
        )

    def forward(self, x, mask=None):
        b, c, h, w = x.size()

        x = x.reshape(b, int((h / self.p) * (w / self.p)), c * self.p * self.p)
        x = self.embeddings(x)

        b, n, e = x.size()

        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, class_token), dim=1)
        x = self.dropout_layer(x + self.positional_encoding)

        for encoder in self.encoders:
            x = encoder(x)

        x = x[:, 0, :]
        self.featuremap = x.detach()
        x = F.log_softmax(self.classifier(self.norm(x)), dim=-1)

        return x
