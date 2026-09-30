import math
from typing import Any

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from style_bert_vits2.models import attentions, commons, modules, monotonic_alignment
from style_bert_vits2.nlp.symbols import NUM_LANGUAGES, NUM_TONES, SYMBOLS


class DurationDiscriminator(nn.Module):  # vits2
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.LSTM = nn.LSTM(
            2 * filter_channels, filter_channels, batch_first=True, bidirectional=True
        )

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(
            nn.Linear(2 * filter_channels, 1), nn.Sigmoid()
        )

    def forward_probability(self, x: torch.Tensor, dur: torch.Tensor) -> torch.Tensor:
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = x.transpose(1, 2)
        x, _ = self.LSTM(x)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        dur_r: torch.Tensor,
        dur_hat: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        x = torch.detach(x)
        if g is not None and hasattr(self, "cond"):
            g = torch.detach(g)
            x = commons.auto_inplace_add(x, self.cond(g))
        x = self.conv_1(x * x_mask)
        x = commons.auto_inplace_relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = commons.auto_inplace_relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, dur)
            output_probs.append(output_prob)

        return output_probs


class TransformerCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
        share_parameter: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        self.wn = (
            # attentions.FFT(
            #     hidden_channels,
            #     filter_channels,
            #     n_heads,
            #     n_layers,
            #     kernel_size,
            #     p_dropout,
            #     isflow=True,
            #     gin_channels=self.gin_channels,
            # )
            None if share_parameter else None
        )

        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    n_layers,
                    n_heads,
                    p_dropout,
                    filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=self.wn,
                    gin_channels=self.gin_channels,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        # メモリ効率化のため、長い系列ではチャンク処理を適用
        seq_len = x.size(2)
        CHUNK_SIZE = 1024
        if seq_len > CHUNK_SIZE:
            return self._chunked_forward(x, x_mask, g, reverse, CHUNK_SIZE)
        else:
            return self._standard_forward(x, x_mask, g, reverse)

    def _standard_forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None,
        reverse: bool,
    ) -> torch.Tensor:
        """従来の Flow 処理の実装"""
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def _chunked_forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None,
        reverse: bool,
        chunk_size: int,
    ) -> torch.Tensor:
        """チャンクごとに処理しメモリ消費の肥大化を抑えた Flow 処理の実装"""
        batch_size, channels, seq_len = x.shape
        overlap_size = 16

        outputs = []

        for start in range(0, seq_len, chunk_size - overlap_size):
            end = min(start + chunk_size, seq_len)
            # チャンク抽出
            chunk_x = x[:, :, start:end]
            chunk_mask = x_mask[:, :, start:end]

            # チャンクごとの Flow 処理
            chunk_output = self._standard_forward(chunk_x, chunk_mask, g, reverse)

            # オーバーラップ処理
            if start == 0:
                # 最初のチャンク
                if end < seq_len:
                    outputs.append(chunk_output[:, :, : -overlap_size // 2])
                else:
                    outputs.append(chunk_output)
            elif end >= seq_len:
                # 最後のチャンク
                outputs.append(chunk_output[:, :, overlap_size // 2 :])
            else:
                # 中間チャンク
                outputs.append(
                    chunk_output[:, :, overlap_size // 2 : -overlap_size // 2]
                )

            # メモリ解放
            del chunk_x, chunk_mask, chunk_output

        # 結合
        result = torch.cat(outputs, dim=2)

        # 長さ調整
        if result.size(2) != seq_len:
            if result.size(2) < seq_len:
                # パディング
                pad_size = seq_len - result.size(2)
                result = torch.nn.functional.pad(result, (0, pad_size))
            else:
                # トリミング
                result = result[:, :, :seq_len]

        return result


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None and hasattr(self, "cond"):
            g = torch.detach(g)
            x = commons.auto_inplace_add(x, self.cond(g))
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = torch.detach(x)
        if g is not None and hasattr(self, "cond"):
            g = torch.detach(g)
            x = commons.auto_inplace_add(x, self.cond(g))
        x = self.conv_1(x * x_mask)
        x = commons.auto_inplace_relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = commons.auto_inplace_relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class Bottleneck(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        super().__init__(c_fc1, c_fc2)


class Block(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = MLP(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(self.norm(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.c_fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # Keep TextEncoder vocabulary size consistent with the caller contract.
        # JP-Extra still uses the standard SYMBOLS set, so detect mismatches early.
        assert n_vocab == len(SYMBOLS), (
            f"JP-Extra TextEncoder expected n_vocab: {len(SYMBOLS)}, got: {n_vocab}"
        )
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.tone_emb = nn.Embedding(NUM_TONES, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        self.language_emb = nn.Embedding(NUM_LANGUAGES, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        self.bert_proj = nn.Conv1d(1024, hidden_channels, 1)

        # Remove emo_vq since it's not working well.
        self.style_proj = nn.Linear(256, hidden_channels)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        g: torch.Tensor | None = None,
        use_fp16: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bert_emb = self.bert_proj(bert).transpose(1, 2)
        style_emb = self.style_proj(style_vec.unsqueeze(1))
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + style_emb
        ) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask, g=g, use_fp16=use_fp16)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock_str: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock_str == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        ch = None
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))  # type: ignore

        assert ch is not None
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(commons.init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None and hasattr(self, "cond"):
            x = commons.auto_inplace_add(x, self.cond(g))

        for i in range(self.num_upsamples):
            x = commons.auto_inplace_leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            assert xs is not None
            x = xs / self.num_kernels
        x = commons.auto_inplace_leaky_relu(x)
        x = self.conv_post(x)
        x = commons.auto_inplace_tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        # print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()  # type: ignore


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(commons.get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = commons.auto_inplace_leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = commons.auto_inplace_leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class WavLMDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self,
        slm_hidden: int = 768,
        slm_layers: int = 13,
        initial_channel: int = 64,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.pre = norm_f(
            Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0)
        )

        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv1d(
                        initial_channel, initial_channel * 2, kernel_size=5, padding=2
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        initial_channel * 2,
                        initial_channel * 4,
                        kernel_size=5,
                        padding=2,
                    )
                ),
                norm_f(
                    nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)
                ),
            ]
        )

        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = commons.auto_inplace_leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels: int, gin_channels: int = 0) -> None:
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = commons.auto_inplace_relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(
        self, L: int, kernel_size: int, stride: int, pad: int, n_convs: int
    ) -> int:
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        n_speakers: int = 256,
        gin_channels: int = 256,
        use_sdp: bool = True,
        n_flow_layer: int = 4,
        n_layers_trans_flow: int = 6,
        flow_share_parameter: bool = False,
        use_transformer_flow: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", True
        )
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        self.enc_gin_channels = 0
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.enc_gin_channels,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                p_dropout,
                n_flow_layer,
                gin_channels=gin_channels,
                share_parameter=flow_share_parameter,
            )
        else:
            self.flow = ResidualCouplingBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                n_flow_layer,
                gin_channels=gin_channels,
            )
        self.sdp = StochasticDurationPredictor(
            hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
        )
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )

        if n_speakers >= 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        else:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        mas_allowed: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, style_vec, g=g
        )
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            if self.use_noise_scaled_mas:
                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )
                neg_cent = neg_cent + epsilon
            # 強制アライメントで決めた音素の範囲 ([b, t_t, t_s] の 0/1) が渡されたら、その外を通れないようにする
            ## maximum_path はマスクを長さの計算にしか使わないので、範囲の外はスコアを大きな負の値にして塞ぐ
            unconstrained_neg_cent = neg_cent
            if mas_allowed is not None:
                neg_cent = neg_cent.masked_fill(mas_allowed == 0, -1e9)

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_alignment.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )
            # 範囲の中で解けなかった発話 (どれかのトークンが 0 フレーム、またはフレームを使い切れていない) だけ、範囲を絞らない MAS に戻す
            if mas_allowed is not None:
                durations = attn.sum(2)
                is_feasible = ((durations + (1 - x_mask)).amin(dim=(1, 2)) >= 1) & (
                    durations.sum(dim=(1, 2)) == y_mask.sum(dim=(1, 2))
                )
                if bool(is_feasible.all()) is False:
                    fallback_attn = (
                        monotonic_alignment.maximum_path(
                            unconstrained_neg_cent, attn_mask.squeeze(1)
                        )
                        .unsqueeze(1)
                        .detach()
                    )
                    attn = torch.where(
                        is_feasible.view(-1, 1, 1, 1), attn, fallback_attn
                    )

        w = attn.sum(2)

        l_length_sdp = self.sdp(x, x_mask, w, g=g)
        l_length_sdp = l_length_sdp / torch.sum(x_mask)

        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        # logw_sdp = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=1.0)
        l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
            x_mask
        )  # for averaging
        # l_length_sdp += torch.sum((logw_sdp - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        l_length = l_length_dp + l_length_sdp

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),  # type: ignore
            (x, logw, logw_),  # , logw_sdp),
            g,
        )

    def infer_input_feature(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        sdp_ratio: float = 0.0,
        y: torch.Tensor | None = None,
        use_fp16: bool = False,
        durations_frames_override: torch.Tensor | None = None,
        durations_frames_override_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generator への入力特徴量（潜在変数）を生成する。通常推論・ストリーミング推論の両方で共通。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                z (latent), y_mask, g (global conditioning), attn (attention), z_p, m_p, logs_p
        """
        # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert)
        # g = self.gst(y)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            assert y is not None
            g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

        # BERT モデルが FP16 でロードされている場合は特徴量も FP16 になるので、明示的に FP32 に変換することでエラーを防ぐ
        if use_fp16 is True:
            bert = bert.float()

        # Encoder は基本 FP32 で実行し、相対位置エンコーディング部分のみ FP16 化
        x, m_p, logs_p, x_mask = self.enc_p(
            x,
            x_lengths,
            tone,
            language,
            bert,
            style_vec,
            g=g,
            use_fp16=use_fp16,
        )

        # 精度クリティカルな部分 (SDP/DP, Flow) は常に FP32 で実行する

        # SDP (Stochastic Duration Predictor) / DP (Duration Predictor)
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)

        # durations_frames_override が指定された場合、指定されたトークンのみ duration を上書きする
        if durations_frames_override is not None:
            override = durations_frames_override
            original_is_floating_point = override.is_floating_point()
            if override.dim() == 1:
                override = override.unsqueeze(0).unsqueeze(0)
            elif override.dim() == 2:
                override = override.unsqueeze(1)
            elif override.dim() != 3:
                raise ValueError(
                    "durations_frames_override must be 1D, 2D, or 3D tensor"
                )

            override = override.to(device=w_ceil.device, dtype=w_ceil.dtype)
            if override.shape != w_ceil.shape:
                raise ValueError(
                    "durations_frames_override shape mismatch. "
                    f"expected: {tuple(w_ceil.shape)}, actual: {tuple(override.shape)}"
                )

            if durations_frames_override_mask is not None:
                mask = durations_frames_override_mask
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.dim() == 2:
                    mask = mask.unsqueeze(1)
                elif mask.dim() != 3:
                    raise ValueError(
                        "durations_frames_override_mask must be 1D, 2D, or 3D tensor"
                    )
                mask = mask.to(device=w_ceil.device, dtype=torch.bool)
                if mask.shape != w_ceil.shape:
                    raise ValueError(
                        "durations_frames_override_mask shape mismatch. "
                        f"expected: {tuple(w_ceil.shape)}, actual: {tuple(mask.shape)}"
                    )
            else:
                # mask が省略された場合、NaN 以外を上書き対象とみなす (float のみ)
                # それ以外の型では、0 より大きい値のみを上書き対象とみなす
                if original_is_floating_point is True:
                    mask = ~torch.isnan(override)
                else:
                    mask = override > 0

            if original_is_floating_point is True:
                invalid_value_mask = ~torch.isfinite(override)
                if torch.any(invalid_value_mask & mask):
                    raise ValueError("durations_frames_override must be finite")

            w_ceil = torch.where(mask, override, w_ceil)
            # 物理的に不正な値の混入を防ぐ
            w_ceil = torch.clamp_min(w_ceil, 0)

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Flow
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        return z, y_mask, g, attn, z_p, m_p, logs_p

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        max_len: int | None = None,
        sdp_ratio: float = 0.0,
        y: torch.Tensor | None = None,
        use_fp16: bool = False,
        durations_frames_override: torch.Tensor | None = None,
        durations_frames_override_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        # Generator 実行前の共通処理
        z, y_mask, g, attn, z_p, m_p, logs_p = self.infer_input_feature(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            style_vec,
            noise_scale,
            length_scale,
            noise_scale_w,
            sdp_ratio,
            y,
            use_fp16,
            durations_frames_override,
            durations_frames_override_mask,
        )

        # Generator (Decoder) のみ全体を FP16 / AMP (Automatic Mixed Precision) で実行
        if use_fp16 is True:
            # z テンソルを Generator の入力用に FP16 に変換
            z_input = (z * y_mask)[:, :, :max_len]
            # デバイスタイプを動的に取得
            device_type = z_input.device.type
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=True,
            ):
                # Generator への入力を FP16 に変換
                o = self.dec(z_input.half(), g=g.half())
        else:
            # FP16 を使わない場合は通常通り実行
            o = self.dec((z * y_mask)[:, :, :max_len], g=g)

        return (o, attn, y_mask, (z, z_p, m_p, logs_p))
