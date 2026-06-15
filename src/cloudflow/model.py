"""Flow-matching UNet with compact ERA5 attention."""

import math

import torch
from physicsnemo.models.diffusion import Conv2d, Linear, UNet, weight_init
from physicsnemo.models.diffusion.song_unet import SongUNet
from physicsnemo.models.diffusion.unet import MetaData
from physicsnemo.registry import ModelRegistry
from torch import nn


def _get_channel_shapes(input_channels: list) -> tuple[int, int, torch.Tensor]:
    pressure_levels = sorted(set(int(ch["level"]) for ch in input_channels if len(ch["level"]) > 0))
    num_atmos_vars = len(
        set(ch["name"][: -len("_" + ch["level"])] for ch in input_channels if len(ch["level"]) > 0)
    )
    num_surf_vars = len(set(ch["name"] for ch in input_channels if ch["level"] == ""))
    return num_atmos_vars, num_surf_vars, torch.Tensor(pressure_levels)


class FourierLevelEmbedding(nn.Module):
    """Fourier embeddings for pressure levels (Aurora-style)."""

    def __init__(self, lower: float, upper: float, d: int):
        super().__init__()
        if d % 2 != 0:
            raise ValueError("The dimensionality must be a multiple of two.")
        self.register_buffer(
            "wavelengths",
            torch.logspace(math.log10(lower), math.log10(upper), d // 2, base=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wavelengths = self.wavelengths.to(x.dtype)
        x = x.ger(2 * torch.pi / wavelengths)
        return torch.cat([x.cos(), x.sin()], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int = 32,
        num_heads: int = 8,
        num_queries: int = 1,
        amp_mode: bool = False,
        fused_conv_bias: bool = False,
    ):
        super().__init__()
        self.q = nn.Parameter(
            weight_init(
                (num_heads, num_queries, latent_dim),
                mode="xavier_uniform",
                fan_in=num_heads * latent_dim,
                fan_out=num_heads * num_queries,
            )
        )
        self.kv = Conv2d(
            in_channels=in_channels,
            out_channels=2 * latent_dim * num_heads,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
        )
        self.proj = Conv2d(
            in_channels=num_heads * latent_dim,
            out_channels=out_channels,
            kernel=1,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
        )
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.amp_mode = amp_mode

    def forward(self, x, attn_mask=None):
        q = self.q.to(x.dtype)
        k, v = (
            torch.permute(
                self.kv(x.flatten(0, 1)).reshape(
                    x.shape[0], x.shape[1], -1, x.shape[-2], x.shape[-1]
                ),
                (0, 3, 4, 1, 2),
            )
            .reshape(
                x.shape[0] * x.shape[3] * x.shape[4],
                self.num_heads,
                x.shape[1],
                self.latent_dim,
                2,
            )
            .unbind(-1)
        )
        attn = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=1 / math.sqrt(self.latent_dim)
        )  # (B*H*W, h, d) — assumes num_queries=1
        out = torch.permute(attn.reshape(x.shape[0], x.shape[3], x.shape[4], -1), (0, 3, 1, 2))
        return self.proj(out)


class FlowAttCompressUNet(UNet):
    """Flow-matching UNet with compact ERA5 cross-attention.

    The model attends over pressure levels of ERA5 atmospheric variables at
    each spatial location, then concatenates the resulting representation with
    surface metadata and the noisy MODIS image before passing through a
    SongUNet backbone.

    At inference time, atmospheric attention is run only on the *unique* ERA5
    cells covered by the MODIS pixels (``compact_representative``), then
    scatter-expanded back to the MODIS grid (``compact_inverse``). Both index
    tensors must be supplied by the caller — they are not derived inside the
    model so that the forward pass remains torch.compile compatible.
    """

    def __init__(
        self,
        img_resolution: int | tuple[int, int],
        img_in_channels: int,
        model_channels: int,
        img_out_channels: int,
        input_channels: list,
        use_fp16: bool = False,
        compressed_input_channels: int = 64,
        att_latent_dim: int = 32,
        att_heads: int = 8,
        att_queries: int = 1,
        version2: bool = False,
        **model_kwargs: dict,
    ):
        super(UNet, self).__init__(meta=MetaData)

        if isinstance(img_resolution, int):
            self.img_shape_x = self.img_shape_y = img_resolution
        else:
            self.img_shape_y = img_resolution[0]
            self.img_shape_x = img_resolution[1]

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.version2 = version2

        self.num_atmos_vars, self.num_surf_vars, self.press_levels = _get_channel_shapes(
            input_channels
        )
        self.total_num_levels = len(self.press_levels)

        N_grid_channels = model_kwargs.get("N_grid_channels", 0)
        unet_in_channels = img_out_channels + compressed_input_channels + N_grid_channels
        if version2:
            unet_in_channels += self.num_surf_vars
        self.model = SongUNet(
            img_resolution=img_resolution,
            in_channels=unet_in_channels,
            out_channels=img_out_channels,
            model_channels=model_channels,
            **model_kwargs,
        )
        self.use_fp16 = use_fp16

        self.atmos_embed = Conv2d(
            in_channels=self.num_atmos_vars,
            out_channels=att_latent_dim,
            kernel=1,
            fused_conv_bias=True,
            amp_mode=self.amp_mode,
            init_mode="xavier_uniform",
        )

        self.multi_head_attention = MultiHeadAttention(
            in_channels=att_latent_dim,
            out_channels=compressed_input_channels,
            latent_dim=att_latent_dim,
            num_heads=att_heads,
            num_queries=att_queries,
            amp_mode=self.amp_mode,
        )

        self.press_level_encoding = FourierLevelEmbedding(
            lower=self.press_levels[0], upper=self.press_levels[-1], d=att_latent_dim
        )
        self.press_level_embed = Linear(att_latent_dim, att_latent_dim, amp_mode=self.amp_mode)

        if not version2:
            self.channel_down = Conv2d(
                in_channels=compressed_input_channels + self.num_surf_vars,
                out_channels=compressed_input_channels,
                kernel=1,
                fused_conv_bias=True,
                amp_mode=self.amp_mode,
                init_mode="xavier_uniform",
            )

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        compact_representative: torch.Tensor = None,
        compact_inverse: torch.Tensor = None,
        level_mask: torch.Tensor = None,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        num_levels = self.total_num_levels
        B, _, H, W = img_lr.shape
        img_lr = img_lr.to(dtype)

        img_surf = img_lr[:, self.num_atmos_vars * num_levels :]
        img_atmos = img_lr[:, : self.num_atmos_vars * num_levels]

        attn_mask = None
        if level_mask is not None:
            attn_mask = level_mask.unsqueeze(0).unsqueeze(0)

        pressure_levels = self.press_levels.to(dtype=dtype, device=x.device)
        press_embed = self.press_level_encoding(pressure_levels)
        press_embed = self.press_level_embed(press_embed)

        if compact_representative is not None:
            z_atmos = self._compact_attention(
                img_atmos,
                compact_representative,
                compact_inverse,
                num_levels,
                press_embed,
                B,
                H,
                W,
                attn_mask=attn_mask,
            )
        else:
            if self.version2:
                img_atmos = (
                    img_atmos.reshape(B, self.num_atmos_vars, num_levels, H, W)
                    .permute(0, 2, 1, 3, 4)
                    .contiguous()
                )
            else:
                img_atmos = img_atmos.reshape(B, num_levels, self.num_atmos_vars, H, W)
            img_atmos_embed = self.atmos_embed(img_atmos.flatten(0, 1)).reshape(
                B, num_levels, -1, H, W
            )
            img_atmos_embed = img_atmos_embed + press_embed[None, :, :, None, None]
            z_atmos = self.multi_head_attention(img_atmos_embed, attn_mask=attn_mask)

        img_lr = torch.cat((z_atmos, img_surf), dim=1)

        if not self.version2:
            img_lr = self.channel_down(img_lr)

        x = torch.cat((x.to(dtype), img_lr), dim=1)
        F_x = self.model(
            x.to(dtype),
            sigma.to(dtype).flatten(),
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead.")

        return F_x.to(torch.float32)

    def _compact_attention(
        self,
        img_atmos: torch.Tensor,
        compact_representative: torch.Tensor,
        compact_inverse: torch.Tensor,
        num_levels: int,
        press_embed: torch.Tensor,
        B: int,
        H: int,
        W: int,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        max_unique = compact_representative.shape[1]
        C = img_atmos.shape[1]

        img_atmos_flat = img_atmos.reshape(B, C, H * W)
        gather_idx = compact_representative.unsqueeze(1).expand(B, C, max_unique)
        compact_atmos = torch.gather(img_atmos_flat, 2, gather_idx)

        if self.version2:
            compact_atmos = (
                compact_atmos.reshape(B, self.num_atmos_vars, num_levels, max_unique)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        else:
            compact_atmos = compact_atmos.reshape(B, num_levels, self.num_atmos_vars, max_unique)

        compact_embed = self.atmos_embed(compact_atmos.flatten(0, 1).unsqueeze(-1)).reshape(
            B, num_levels, -1, max_unique
        )
        compact_embed = compact_embed + press_embed[None, :, :, None]

        z_compact = self.multi_head_attention(
            compact_embed.unsqueeze(-1), attn_mask=attn_mask
        ).squeeze(-1)

        out_channels = z_compact.shape[1]
        expand_idx = compact_inverse.unsqueeze(1).expand(B, out_channels, H * W)
        return torch.gather(z_compact, 2, expand_idx).reshape(B, out_channels, H, W)


def _register() -> None:
    """Register ``FlowAttCompressUNet`` with physicsnemo's model registry.

    Checkpoints serialise the class under its original corrdiff module path
    (``models.unet_preprocess``), which does not exist here. Registering the
    class by name lets ``Module.from_checkpoint`` resolve it from the registry
    instead of importing that missing module.
    """
    registry = ModelRegistry()
    if "FlowAttCompressUNet" not in registry.list_models():
        registry.register(FlowAttCompressUNet, name="FlowAttCompressUNet")


_register()
