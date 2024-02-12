# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import get_activation
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from diffusers.models.lora import LoRACompatibleLinear

from .attention_processor import Attention

import math

@maybe_allow_in_graph
class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x, objs):
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.

        if attention_mask is not None and not isinstance(attention_mask, list):
            if attention_mask is not None and hidden_states.shape[1] != attention_mask.shape[-1]:
                tmp = attention_mask.clone()
                scale_factor = int(math.sqrt(attention_mask.shape[-1] // hidden_states.shape[1]))
                try:
                    tmp = tmp.reshape(tmp.shape[0], 40, 72)
                except:
                    try:
                        tmp = tmp.reshape(tmp.shape[0], 32, 32) # MSR-VTT
                    except:
                        tmp = tmp.reshape(tmp.shape[0], 96, 96)
                tmp = tmp[:, ::scale_factor, ::scale_factor]
                tmp = tmp.reshape(tmp.shape[0], 1, -1)
                attention_mask = tmp
            
            if attention_mask is not None:
                tmp = attention_mask.clone()
                tmp = tmp.view(tmp.shape[0], -1,1)/(-10000)
                tmp = (1-tmp)        
                orig_attn_mask = attention_mask.clone()
            else: 
                # tmp = 0
                tmp =1
                orig_attn_mask = None

            if attention_mask is not None and 'make_2d_attention_mask' in kwargs and kwargs['make_2d_attention_mask'] == True:
                # We broadcast and take element wise AND. Note that addition is equivalent to AND here, since we are dealing with -10000 and 0. 
                attention_mask_2d = attention_mask + attention_mask.permute(0,2,1) 
                # Get it back to original range. This step is optional tbh
                attention_mask_2d = torch.where(attention_mask_2d < 0., -10000, 0).type(attention_mask.dtype)
                
                if 'block_diagonal_attention' in kwargs and kwargs['block_diagonal_attention'] == True:
                    tmp_attention = torch.where(attention_mask < 0., 0., -10000.) # allow background
                    tmp_attention = tmp_attention + tmp_attention.permute(0,2,1)
                    tmp_attention = torch.where(tmp_attention < 0., -10000, 0)
                    attention_mask_2d = attention_mask_2d * tmp_attention
                    attention_mask_2d = torch.where(attention_mask_2d.abs() < 1.,0., -10000.).type(attention_mask.dtype)
                attention_mask = attention_mask_2d
        
        
        # Multiple objects
        elif attention_mask is not None and isinstance(attention_mask, list):
            if hidden_states.shape[1] != attention_mask[0].shape[-1]:
                new_attention_mask = []
                for attn_mask in attention_mask:
                    tmp = attn_mask.clone()
                    scale_factor = int(math.sqrt(attn_mask.shape[-1] // hidden_states.shape[1]))
                    try:
                        tmp = tmp.reshape(tmp.shape[0], 40, 72)
                    except:
                        tmp = tmp.reshape(tmp.shape[0], 32, 32)
                    tmp = tmp[:, ::scale_factor, ::scale_factor]
                    tmp = tmp.reshape(tmp.shape[0], 1, -1)
                    new_attention_mask.append(tmp)
                attention_mask = new_attention_mask
                
            orig_attn_mask = []
            for attn_mask in attention_mask:
                tmp = attn_mask.clone()

                tmp = tmp.view(tmp.shape[0], -1,1)/(-10000)
                tmp = (1-tmp)

                orig_attn_mask.append(attn_mask.clone())           


            if 'make_2d_attention_mask' in kwargs and kwargs['make_2d_attention_mask'] == True:
                # We broadcast and take element wise AND. Note that addition is equivalent to AND here, since we are dealing with -10000 and 0. 
                attn_mask_2d = []
                for attn_mask in attention_mask:
                    attention_mask_2d = attn_mask + attn_mask.permute(0,2,1) 
                    # Get it back to original range. This step is optional tbh
                    attention_mask_2d = torch.where(attention_mask_2d < 0., -10000, 0).type(attn_mask.dtype)
                    attn_mask_2d.append(attention_mask_2d)
                attention_mask_2d = torch.prod(torch.stack(attn_mask_2d, dim=0), dim=0)
                attention_mask_2d = torch.where(attention_mask_2d.abs() < 1.,0., -10000.).type(attn_mask.dtype)
                if 'block_diagonal_attention' in kwargs and kwargs['block_diagonal_attention'] == True:
                    tmp_attention = torch.where(torch.prod(torch.stack(attention_mask,dim=0),dim=0).abs() < 1., -10000., 0.) # Check this well
                    tmp_attention = tmp_attention + tmp_attention.permute(0,2,1)
                    tmp_attention = torch.where(tmp_attention < 0., -10000, 0)
                    attention_mask_2d = attention_mask_2d * tmp_attention
                    attention_mask_2d = torch.where(attention_mask_2d.abs() < 1.,0., -10000.).type(attention_mask_2d.dtype)
                attention_mask = attention_mask_2d                                 

        else:
            tmp = 1
            orig_attn_mask = None            
        
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        
        # breakpoint()
            
        ## self-attention amongst fg
        attn_output = self.attn1(
            norm_hidden_states, # + tmp,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        

        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        if attention_mask is not None:
            tmp = 1-tmp

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
        # 2.5 ends

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states*tmp, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states*tmp)
            )

            
            if encoder_attention_mask is None:
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )

            if encoder_attention_mask is not None:  # Encoder attention mask is not None
                
                if 'block_diagonal_attention' in kwargs and kwargs['block_diagonal_attention'] == True:
                    
                    if not isinstance(orig_attn_mask, list):
                        orig_attn_mask = torch.where(orig_attn_mask < 0., 0., -10000.).type(orig_attn_mask.dtype).to(orig_attn_mask.device)
                        encoder_attention_mask_2d = encoder_attention_mask + orig_attn_mask.permute(0,2,1)
                        encoder_attention_mask_2d = torch.where(encoder_attention_mask_2d < 0., -10000, 0).type(encoder_attention_mask.dtype)

                        inverted_encoder_attention_mask = torch.where(encoder_attention_mask < 0., 0., -10000.).type(encoder_attention_mask.dtype)
                        inverted_encoder_attention_mask[:,:,0] = -10000  # CLS token

                        inverted_orig_mask = torch.where(orig_attn_mask < 0., 0., -10000.).type(orig_attn_mask.dtype)
                        inverted_encoder_attention_mask_2d = inverted_encoder_attention_mask + inverted_orig_mask.permute(0,2,1) 
                        
                        encoder_attention_mask_2d = encoder_attention_mask_2d * inverted_encoder_attention_mask_2d
                        encoder_attention_mask_2d = torch.where(encoder_attention_mask_2d.abs() < 1.,0., -10000.).type(encoder_attention_mask.dtype)

                        encoder_attention_mask = encoder_attention_mask_2d
                    else:
                        orig_attn_mask = [torch.where(orig_attn_mask_ < 0., 0., -10000.).type(orig_attn_mask_.dtype).to(orig_attn_mask_.device) for orig_attn_mask_ in orig_attn_mask]
                        encoder_attention_mask_2d = [encoder_attention_mask_ + orig_attn_mask_.permute(0,2,1) for encoder_attention_mask_, orig_attn_mask_ in zip(encoder_attention_mask, orig_attn_mask)]
                        encoder_attention_mask_2d = [torch.where(encoder_attention_mask_2d_ < 0., -10000, 0).type(encoder_attention_mask_2d_.dtype) for encoder_attention_mask_2d_ in encoder_attention_mask_2d]

                        inverted_encoder_attention_mask = torch.where(torch.sum(torch.stack(encoder_attention_mask, dim=0),dim=0) < 0., 0., -10000.).type(encoder_attention_mask[0].dtype)    
                        inverted_encoder_attention_mask[:,:,0] = -10000  # CLS token

                        inverted_orig_mask = torch.where(torch.sum(torch.stack(orig_attn_mask,dim=0),dim=0) < 0., 0., -10000.).type(orig_attn_mask[0].dtype)
                        inverted_encoder_attention_mask_2d = inverted_encoder_attention_mask + inverted_orig_mask.permute(0,2,1) 

                        encoder_attention_mask_2d = torch.where(torch.sum(torch.stack(encoder_attention_mask_2d, dim=0), dim=0) < 0., -10000., 0.)
                        encoder_attention_mask_2d = encoder_attention_mask_2d * inverted_encoder_attention_mask_2d
                        encoder_attention_mask_2d = torch.where(encoder_attention_mask_2d.abs() < 1.,0., -10000.).type(encoder_attention_mask[0].dtype)

                        encoder_attention_mask = encoder_attention_mask_2d
                                                
                    norm_hidden_states = (
                        self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                    )
                    ## cross-attention amongst bg
                    attn_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )

                    del encoder_attention_mask_2d, inverted_encoder_attention_mask, inverted_encoder_attention_mask_2d, inverted_orig_mask, orig_attn_mask, attention_mask_2d, tmp_attention
                    torch.cuda.empty_cache()

                    hidden_states = attn_output + hidden_states

                else:
                    norm_hidden_states2 = (
                        self.norm2(hidden_states*(1-tmp), timestep) if self.use_ada_layer_norm else self.norm2(hidden_states*(1-tmp))
                    )
                    encoder_attention_mask2 = torch.where(encoder_attention_mask < 0., 0., -10000.).type(encoder_attention_mask.dtype).to(encoder_attention_mask.device)
                    encoder_attention_mask2[:, :, 0] = -10000
                    attn_output2 = self.attn2(
                        norm_hidden_states2,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask2,
                        **cross_attention_kwargs,
                    )

                    hidden_states = attn_output*tmp + attn_output2*(1-tmp)+ hidden_states
            else:
                hidden_states = attn_output*tmp + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice, scale=lora_scale)
                    for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(LoRACompatibleLinear(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states, scale: float = 1.0):
        for module in self.net:
            if isinstance(module, (LoRACompatibleLinear, GEGLU)):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = LoRACompatibleLinear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states, scale: float = 1.0):
        hidden_states, gate = self.proj(hidden_states, scale).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Module):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()

        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, timestep, class_labels, hidden_dtype=None):
        emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaGroupNorm(nn.Module):
    """
    GroupNorm layer modified to incorporate timestep embeddings.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x, emb):
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x
