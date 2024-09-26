#----------------------------------------------------------------------------#
#implement of text_encoder
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class ClapDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is a slightly
    refactored version of the `SwinDropPath` implementation.
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states):
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states

        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)

        random_tensor = keep_prob + torch.rand(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        random_tensor.floor_()  # binarize
        output = hidden_states.div(keep_prob) * random_tensor
        return output

class mlp(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(input_dim, latent_dim)
        self.linear2 = nn.Linear(latent_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x 




class MultiheadAttn(nn.Module):
    def __init__(self, tConfig):
        super().__init__()
        self.nhead = tConfig.n_head

        self.query = nn.Linear(tConfig.d_model, tConfig.d_model)
        self.key = nn.Linear(tConfig.d_model ,tConfig.d_model)
        self.value = nn.Linear(tConfig.d_model, tConfig.d_model)
        self.LayerNorm = nn.LayerNorm(tConfig.d_model)
        self.output = nn.Linear(tConfig.d_model, tConfig.d_model)
    
    def forward(self, q, k, v, attn_mask):
        B, L, C = q.shape
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        q, k, v = x.split(self.n_embd, dim = 2)

        q = q.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 /math.sqrt(k.size(-1)))
        attn = torch.masked_fill(attn, attn_mask, float('-inf'))
        attn = F.softmax(attn, dim = -1)


        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(B, L, C)
        x = self.output(x)
        x = self.LayerNorm(x)
        return x


class tBlock(nn.Module):
    def __init__(self, tConfig):
        super().__init__()
        self.attention = MultiheadAttn(tConfig)
        self.intermedate = nn.Linear(tConfig.d_model, 3 * tConfig.d_model)
        self.output = nn.ModuleDict(dict(dense = nn.Linear(3* tConfig.d_model, tConfig.d_model),
                                         LayerNorm = nn.LayerNorm(tConfig.d_model)))

    def forward(self, input):
        x = self.attention(input)
        x = self.intermedate(x)
        x = self.output.dense(self.output.layerNorm(x))

class Text_encoder(nn.Module):
    def __init__(self, tConfig, device):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            'word_embeddings': nn.Embedding(50265, tConfig.d_model),
            'position_embeddings': nn.Embedding(tConfig.vocab_length, tConfig.d_model),
            'token_type_embeddings': nn.Embedding(1, tConfig.d_model),
            'LayerNorm': nn.LayerNorm(tConfig.d_model)
        })
        # Add parameters directly as attributes of the module
        self.embeddings.position_ids = nn.Parameter(torch.Tensor(1, tConfig.vocab_length).to(device))
        self.embeddings.token_type_ids = nn.Parameter(torch.Tensor(1, tConfig.vocab_length).to(device))
        
        self.encoder = nn.ModuleList(tBlock(tConfig) for _ in range (tConfig.tenc_layers))
        self.pooler = nn.Linear(tConfig.d_model, tConfig.d_model)

    def forward(self, tokens):
        pos = self.embeddings.position_ids
        type_id = self.embeddings.token_type_ids
        w_emdb = self.embeddings.word_embeddings(tokens)
        t_embd = self.embeddings.token_type_embeddings(type_id)
        pos_embd = self.embeddings.position_embeddings(pos)
        input_embd = w_emdb + t_embd + pos_embd
        input_embd = self.embeddings.LayerNorm(input_embd)

        output = self.encoder(input_embd)
        output = output + self.pooler(output)
        return output


class Text_projection(nn.Module):
    def __init__(self, Config):
        super().__init__()
        self.linear1 = nn.Linear(Config.vocab_size, Config.d_model)
        self.linear2 = nn.Linear(Config.vocab_size, Config.vocab_size)
    
    def forward(self, t_embd):
        t_embd = t_embd + self.linear1(t_embd)
        output = t_embd + self.linear2(t_embd)
        return output
    
#--------------------------------------------------------------------------------------#
#implement of audio encoder
from typing import List

#copy paste from https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/htsat.py
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
    
class patch_embed(nn.Module):  #2d image to patch embedding, i think here we just need to implement a simple embd?
    def __init__(self, aConfig):      
        super().__init__()
        self.proj = nn.Conv2d(aConfig.in_channels, aConfig.d_model, kernel_size=aConfig.patchsize, stride=aConfig.patchsize)
        self.norm = nn.LayerNorm(aConfig.d_model)
    
    def forward(self, x):
        x = self.proj(x)  # Shape: (batch_size, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        x = self.norm(x)  # Shape: (batch_size, num_patches, embed_dim)
        return x


class audio_encoder(nn.Module):
    def __init__(self, aConfig):
        super().__init__()
        self.num_layers = len(aConfig.layer_list)
        self.norm = nn.LayerNorm(1024)
        self.spec_size = 256
        self.freq_ratio = 4
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers - 1):
            layer = basiclayers(aConfig, num_block = aConfig.layer_list[i], 
                                spatial_shape = [dim //((2 ** i) * aConfig.patchsize) for dim in aConfig.spatial_resolution], 
                                dim = aConfig.d_model * (2 ** i), not_last = True, num_heads = aConfig.num_heads[i])

            self.layers.append(layer)
        self.layers.append(basiclayers(aConfig, num_block = aConfig.layer_list[self.num_layers - 1], 
                                spatial_shape = [dim //((2 ** (self.num_layers - 1)) * aConfig.patchsize) for dim in aConfig.spatial_resolution], 
                                dim = aConfig.d_model * (2 ** (self.num_layers - 1)), not_last = False, num_heads = aConfig.num_heads[-1]))
        self.patch_embed = patch_embed(aConfig)
        self.batch_norm = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    
    def reshape_mel2img(self, normalized_input_features):
        """
        The input is 4 normalized log mel spectrograms. It is reshape to the common shape of images. Each channel
        should represent 1 of the 4 crops of the spectrogram. For more details, refer to the [`ClapFeatureExtractor`].
        """
        _, _, time_length, freq_length = normalized_input_features.shape

        spec_width = int(self.spec_size * self.freq_ratio)
        spec_heigth = self.spec_size // self.freq_ratio
        print(f"lets see whats going on: {spec_width}, {spec_heigth}")

        if time_length > spec_width or freq_length > spec_heigth:
            raise ValueError("the wav size should be less than or equal to the swin input size")

        # to avoid bicubic zero error
        if time_length < spec_width:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (spec_width, freq_length), mode="bicubic", align_corners=True
            )
        if freq_length < spec_heigth:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (time_length, spec_heigth), mode="bicubic", align_corners=True
            )

        batch, channels, time, freq = normalized_input_features.shape

        # batch_size, channels, spec_width, spec_heigth --> batch_size, channels, spec_heigth * freq_ratio, spec_width // freq_ratio
        normalized_input_features = normalized_input_features.reshape(
            batch, channels * self.freq_ratio, time // self.freq_ratio, freq
        )
        normalized_input_features = normalized_input_features.permute(0, 1, 3, 2).contiguous()
        normalized_input_features = normalized_input_features.reshape(
            batch, channels, freq * self.freq_ratio, time // self.freq_ratio
        )

        return normalized_input_features

    def forward(self, src):
        src = src.permute(0, 3, 2, 1)
        src = self.batch_norm(src)
        src = src.permute(0, 3, 2, 1)

        src = self.reshape_mel2img(src)
        output = self.patch_embed(src)

        for _, layer in enumerate(self.layers):
            output = layer(output)
        
        output = self.norm(output)
        output = output.permute(0, 2, 1)                             #permute to do pooling on desire dimension
#
        output = self.avgpool(output)
        output = output.permute(0, 2, 1)

        return output

class basiclayers(nn.Module):
    def __init__(self, aConfig, num_block, spatial_shape, dim, not_last, num_heads):
        super().__init__()
        self.num_block = num_block
        self.dim = dim
        self.not_last = not_last
        self.blocks = nn.ModuleList()
        self.spatial_shape = spatial_shape
        self.num_heads = num_heads
        self.window_size = aConfig.window_size
        for i in range(self.num_block):
            block = Block(aConfig, spatial_shape = spatial_shape, shift_size=0 if (i % 2 == 0) else self.window_size //2 ,      #hardcode here, the shift size should be windowsize//2
                          dim = self.dim, num_heads = self.num_heads)
            self.blocks.append(block)
        if not_last:
            self.downsample = downsample(spatial_shape, dim = self.dim)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.not_last:
            x = self.downsample(x)
        return x
    

class Block(nn.Module):  # Swin transformer block
    def __init__(self, aConfig, spatial_shape, shift_size = 0, dim = 256, num_heads = 8):
        super().__init__()
        self.spatial_shape = spatial_shape
        if aConfig.window_size >= min(spatial_shape):
            shift_size = 0
            self.window_size = min(spatial_shape)
        else:
            self.window_size = aConfig.window_size
        self.shift_size = shift_size
        self.patchsize = aConfig.patchsize
        self.dim = dim
        self.num_heads = num_heads
        self.drop_path = ClapDropPath(aConfig.drop_path_rate) if aConfig.drop_path_rate > 0.0 else nn.Identity()

        self.layernorm_before = nn.LayerNorm(self.dim)
        self.attention = WindowAttention(aConfig, spatial_shape = self.spatial_shape, dim = self.dim, 
                                         window_size = self.window_size, num_heads = self.num_heads)
        self.layernorm_after = nn.LayerNorm(self.dim)
        self.intermediate = Intermediate(self.dim)
        self.output = Output(self.dim*4, self.dim)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.spatial_shape
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # Pre-attention norm
        B, L, C = x.shape
        H, W = self.spatial_shape
        shortcut = x

        x = self.layernorm_before(x)
        assert H * W == L , f"spatial_shape is unaligned with length of features, expect shape:{H} * {W}, acutal input: {L}"
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        window_x = window_partition(shifted_x, self.window_size)
        
        attn_output, attn_weights = self.attention(window_x, self.attn_mask)
        
        shifted_x = window_reverse(attn_output, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.reshape(B, L, C)
        x = shortcut + x
        # Post-attention norm
        x = self.layernorm_after(x)
        
        # Intermediate fully connected layer with GELU activation and residual connection
        intermediate_output = self.intermediate(x)
        x = x + self.output(intermediate_output)
        
        
        return x

class downsample(nn.Module):
# Example usage
    def __init__(self, spatial_shape, dim):
        super().__init__()
        self.dim = dim
        self.spatial_shape = spatial_shape
        self.reduction = nn.Linear(4 * self.dim, 2 * self.dim, bias=False)
        self.norm = nn.LayerNorm(4 * self.dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.spatial_shape
        B, L, C = x.shape
        
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class WindowAttention(nn.Module):
    def __init__(self, config, spatial_shape, dim, window_size, num_heads):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim = dim

        self.query = nn.Linear(self.dim, self.dim)
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.output = Output(self.dim, self.dim)
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * 8 - 1) * (2 * 8 - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        #relative position encoding setup
        coords_h = torch.arange(8)
        coords_w = torch.arange(8)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += 8 - 1  # shift to start from 0
        relative_coords[:, :, 1] += 8 - 1
        relative_coords[:, :, 0] *= 2 * 8 - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop =nn.Dropout(0.1)

    def forward(self, x, attn_mask = None):
        B, H, W, C = x.shape
        N = H * W 
        x = x.reshape(B, N, C) 

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        # Apply relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            8 * 8, 8 * 8, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        scores = scores + relative_position_bias.unsqueeze(0)

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            scores = scores.view(B // nW, nW, self.num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
            scores = scores.view(-1, self.num_heads, N, N)
            scores = F.softmax(scores, dim=-1)
        else:
            scores = F.softmax(scores, dim=-1)

        scores = self.attn_drop(scores)
        

        output = torch.matmul(scores, v)
        output = output.reshape(B, H, W, C)
        return output, scores

class Intermediate(nn.Module):
    def __init__(self, dim):
        super(Intermediate, self).__init__()
        self.dim = dim
        self.dense = nn.Linear(self.dim, self.dim * 4)
        self.act = F.gelu

    def forward(self, x):
        return self.act(self.dense(x))

class Output(nn.Module):
    def __init__(self, dim1, dim2):
        super(Output, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dense = nn.Linear(self.dim1, self.dim2)

    def forward(self, x):
        return self.dense(x)


def build_text_encoder(Config):
    return Text_encoder(Config)

def build_audio_encoder(Config):
    return audio_encoder(Config)

def build_mlp(input_dim, latent_dim, output_dim):
    return mlp(input_dim, latent_dim, output_dim)
