import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pscan import pscan

@dataclass
class MambaConfig:
    d_model: int  #  D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 2  #  N in paper/comments 2 69%
    expand_factor: int = 2  #  E in paper/comments
    d_conv: int = 2

    dt_min: float = 0.01
    dt_max: float = 0.1
    dt_init: str = "random"  #  "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True  #  use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        down_blocks = []
        use_batchnorm = True
        nch = 1
        powf = 3
        max_powf = 8
        insz = 1024
        minsz = 8
        nblocks = int(math.log(float(insz) / float(minsz), 2))
        self.leakrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.n_filters = [nch] + [2 ** min((i + powf), max_powf) for i in range(nblocks)]
        print(self.n_filters)
        for i in range(len(self.n_filters) - 1):
            down_blocks.extend(self.down_block(self.n_filters[i], self.n_filters[i + 1], use_batchnorm))
        self.down = nn.Sequential(*down_blocks)

        config = MambaConfig(d_model=1, n_layers=2)
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        config1 = MambaConfig(d_model=8, n_layers=1)
        self.layers1 = nn.ModuleList([ResidualBlock(config1) for _ in range(config1.n_layers)])
        config2 = MambaConfig(d_model=32, n_layers=3)
        self.layers2 = nn.ModuleList([ResidualBlock(config2) for _ in range(config2.n_layers)])
        nemb = 1
        cur_sz = int(insz / (2 ** (len(self.n_filters) - 1)))
        inter_ch = int(abs(self.n_filters[-1] + nemb) / 2)
        block1 = []
        block1.append(nn.Conv1d(1, 2, kernel_size=4, stride=2, padding=1))
        block1.append(nn.BatchNorm1d(2, momentum=0.01))
        block1.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        block1.append(nn.Conv1d(2, 4, kernel_size=4, stride=2, padding=1))
        block1.append(nn.BatchNorm1d(4, momentum=0.01))
        block1.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        block1.append(nn.Conv1d(4, 8, kernel_size=4, stride=2, padding=1))
        block1.append(nn.BatchNorm1d(8, momentum=0.01))
        block1.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.block1 = nn.Sequential(*block1)
        self.dropout = nn.Dropout(p=0.05)

        block2 = []

        block2.append(nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=1))
        block2.append(nn.BatchNorm1d(16, momentum=0.01))
        block2.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        block2.append(nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1))
        block2.append(nn.BatchNorm1d(32, momentum=0.01))
        block2.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.block2 = nn.Sequential(*block2)
        block3 = []
        block3.append(nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1))
        block3.append(nn.BatchNorm1d(64, momentum=0.01))
        block3.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        block3.append(nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1))
        block3.append(nn.BatchNorm1d(128, momentum=0.01))
        block3.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        block3.append(nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1))
        block3.append(nn.BatchNorm1d(128, momentum=0.01))
        block3.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.block3 = nn.Sequential(*block3)
        config3 = MambaConfig(d_model=256, n_layers=2)
        self.layers3 = nn.ModuleList([ResidualBlock(config3) for _ in range(config3.n_layers)])
        self.fc1 = nn.Linear(512, 256)  # 第一个全连接层
        self.fc2 = nn.Linear(256, 1)  # 第二个全连接层

    def forward(self, x):
        x = F.interpolate(x, size=1024, mode='linear', align_corners=True)
        x = self.block1(x)
        x = torch.transpose(x, 1, 2)
        for layer in self.layers1:
            x = layer(x)
        x = torch.transpose(x, 1, 2)
        x = self.block2(x)
        x = torch.transpose(x, 1, 2)
        for layer in self.layers2:
            x = layer(x)
        x = torch.transpose(x, 1, 2)
        x = self.block3(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 输出层

        return x

    def step(self, x, caches):
        #  x : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        #  y : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

    def down_block(self, inch, outch, use_batchnorm):
        blocks = []
        # create down block
        blocks.append(nn.Conv1d(inch, outch, kernel_size=4, stride=2, padding=1))
        blocks.append(nn.BatchNorm1d(outch, momentum=0.01))
        blocks.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        blocks.append(nn.Conv1d(outch, outch, kernel_size=3, stride=1, padding=1))
        blocks.append(nn.BatchNorm1d(outch, momentum=0.01))
        blocks.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        return blocks


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        #  x : (B, L, D)

        #  output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs: (B, ED, d_conv-1)

        #  output : (B, D)
        #  cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        #  projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        #  x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  #  (B, L, ED), (B, L, ED)

        #  x branch
        x = x.transpose(1, 2)  #  (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  #  (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, L, D)

        return output

    def ssm(self, x):
        #  x : (B, L, ED)

        #  y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  #  (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs : (B, ED, d_conv-1)

        #  y : (B, D)
        #  cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  #  (B, ED), (B, ED)

        #  x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  #  (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  #  (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        #  x : (B, ED)
        #  h : (B, ED, N)

        #  y : (B, ED)
        #  h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  #  (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)

        h = deltaA * h + BX  #  (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        #  todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
