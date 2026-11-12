import math
import torch
import torch.hub
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import AutoConfig, AutoModel
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x

class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionMixer(d_model=dim,
                                          d_state=8,
                                          d_conv=3,
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class MambaVisionLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 transformer_blocks=[],
                 ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                counter=i,
                transformer_blocks=transformer_blocks,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                layer_scale=layer_scale)
            for i in range(depth)])
        self.transformer_block = True
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        return x

class ConvTransposeUpsample(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4):
        super().__init__()
        if isinstance(input_resolution, int):
            self.input_resolution = (input_resolution, input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution
        else:
            raise TypeError("input_resolution should be tuple or int.")
        self.dim = dim
        self.expand = nn.ConvTranspose2d(dim, int(dim // dim_scale), kernel_size=2, stride=2)

    def forward(self, x):
        h, w = self.input_resolution
        x = rearrange(x, "n (h w) c -> n c h w", h=h, w=w)
        x = self.expand(x)
        x = rearrange(x, "n c h w -> n (h w) c")
        return x

class MambaLayerUp(nn.Module):
    def __init__(
            self,
            dim,
            input_resolution,
            depth,
            num_heads,
            window_size,
            upsample=True,
            dim_scale=4,
            drop_path=0.,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = (
            input_resolution if isinstance(input_resolution, tuple) else (input_resolution, input_resolution)
        )
        self.upsample_flag = upsample

        self.blocks = nn.Sequential(
            MambaVisionLayer(
                dim=dim,
                depth=depth,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=4.0,
                drop_path=drop_path,
                transformer_blocks=list(range(depth // 2 + 1, depth)) if depth % 2 != 0 else list(
                    range(depth // 2, depth)),
            )
        )

        if upsample:
            self.upsample = ConvTransposeUpsample(input_resolution, dim=dim, dim_scale=dim_scale)
        else:
            self.upsample = None

    def forward(self, x):
        H, W = self.input_resolution
        assert x.shape[1] == H * W, f"Expected {H * W} tokens, got {x.shape[1]}"

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.blocks(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        if self.upsample is not None:
            x = self.upsample(x)
        return x

class MambaDecoder(nn.Module):
    def __init__(self, in_resolution, in_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=[8, 8, 8, 8], drop_path_rate=0., crop_size=256):
        super().__init__()
        self.num_layers = len(depths)
        self.crop_size = crop_size
        self.layers_up = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(self.num_layers):
            fused_dim = int(in_dim * 2 ** (self.num_layers - i_layer))
            current_res = in_resolution // (2 ** (self.num_layers - 1 - i_layer))

            layer = MambaLayerUp(
                dim=fused_dim,
                input_resolution=current_res,
                depth=depths[(self.num_layers - 1 - i_layer)],
                num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                window_size=window_size[i_layer],
                upsample=True,
                dim_scale=4,
                drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                    depths[:(self.num_layers - 1 - i_layer) + 1])]
            )
            self.layers_up.append(layer)

        self.batch_size = 16
        self.up_x2 = ConvTransposeUpsample(2 * in_resolution, in_dim // 2, dim_scale=4)
        self.output = nn.Conv2d(in_dim // 8, 3, 1, bias=False)

    def forward(self, x, fm_list):
        x = rearrange(x, "N C H W -> N (H W) C")
        fm_list = [
            rearrange(f, "N C H W -> N (H W) C")
            for f in fm_list
        ]

        n = x.shape[0]
        outputs = []
        for i in range(0, n, self.batch_size):
            end = min(i + self.batch_size, n)
            x_batch = x[i:end]
            fm_batch = [fm[i:end] for fm in fm_list]

            for j in range(self.num_layers):
                skip_feat = fm_batch[self.num_layers - 1 - j]
                x_batch = torch.cat([x_batch, skip_feat], dim=-1)
                x_batch = self.layers_up[j](x_batch)

            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)
        x = self.up_x2(x)
        x = rearrange(x, "n (HC WC) C -> n C HC WC", HC=self.crop_size, WC=self.crop_size)
        x = self.output(x)
        return x

class MHFNet(AbstractModel):
    def __init__(
            self,
            channels_last: bool = True,
            target_size: int = 256,
            type='S'
    ):
        super().__init__()
        self.channels_last = channels_last
        self.target_size = target_size
        self.batch_size = 16

        self._initialize_weights()
        if type == 'T':
            config = AutoConfig.from_pretrained('nvidia/MambaVision-T-1K', trust_remote_code=True, cache_dir='./')
            self.filters = [80, 160, 320, 640]
            self.num_heads = [2, 4, 8, 16]
            self.window_size = [8, 8, 14, 7]
            self.depths = [1, 3, 8, 4]
            self.drop_path_rate = 0.2
        elif type == 'S':
            config = AutoConfig.from_pretrained('nvidia/MambaVision-S-1K', trust_remote_code=True, cache_dir='./')
            self.filters = [96, 192, 384, 768]
            self.num_heads = [2, 4, 8, 16]
            self.window_size = [8, 8, 14, 7]
            self.depths = [3, 3, 7, 5]
            self.drop_path_rate = 0.2
        elif type == 'B':
            config = AutoConfig.from_pretrained('nvidia/MambaVision-B-1K', trust_remote_code=True, cache_dir='./')
            self.filters = [128, 256, 512, 1024]
            self.num_heads = [2, 4, 8, 16]
            self.window_size = [8, 8, 14, 7]
            self.depths = [3, 3, 10, 5]
            self.drop_path_rate = 0.3
        else:
            config = AutoConfig.from_pretrained('nvidia/MambaVision-L-1K', trust_remote_code=True, cache_dir='./')
            self.filters = [196, 392, 784, 1568]
            self.num_heads = [4, 8, 16, 32]
            self.window_size = [8, 8, 14, 7]
            self.depths = [3, 3, 10, 5]
            self.drop_path_rate = 0.3
        self.encoder = AutoModel.from_config(config, trust_remote_code=True)

        self.bottleneck = nn.Sequential(
            *[
                MambaVisionLayer(
                    dim=self.filters[-1],
                    depth=2,
                    num_heads=8,
                    window_size=8,
                    mlp_ratio=4.0,
                    transformer_blocks=list(range(2 // 2 + 1, 2)) if 2 % 2 != 0 else list(range(2 // 2, 2)),
                )
                for _ in range(2)
            ]
        )

        self.decoder = MambaDecoder(in_resolution=self.target_size // 4, in_dim=self.filters[0], depths=self.depths,
                                    num_heads=self.num_heads, window_size=self.window_size,
                                    drop_path_rate=self.drop_path_rate, crop_size=self.target_size)

    def forward(self, x):
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x_skip = x

        _, enc_results = self.encoder(x)

        feat = self.bottleneck(enc_results[-1])

        output = self.decoder(feat, enc_results)


        output = output + x_skip

        return output
