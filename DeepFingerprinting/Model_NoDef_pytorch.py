import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dSame(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, *, bias: bool = False, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, bias=bias, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_pad = self.dilation * (self.kernel_size - 1)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, pool_size: int, pool_stride: int, use_elu: bool, dropout_p: float, bn_momentum: float = 0.01, ) -> None:
        super().__init__()
        self.conv1 = Conv1dSame(in_ch, out_ch, kernel_size, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch, momentum=bn_momentum)
        self.act1 = nn.ELU() if use_elu else nn.ReLU(inplace=True)

        self.conv2 = Conv1dSame(out_ch, out_ch, kernel_size, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch, momentum=bn_momentum)
        self.act2 = nn.ELU() if use_elu else nn.ReLU(inplace=True)

        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        self.drop = nn.Dropout(p=dropout_p)

        self._init_weights(use_elu=use_elu)

    def _init_weights(self, *, use_elu: bool) -> None:
        # He initialization adapted to the nonlinearity
        nonlin = "relu" if not use_elu else "relu"  # ELU often uses the same gain
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlin)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        return x


class DFNoDefNet(nn.Module):
    def __init__(self, num_classes = 95) -> None:
        super().__init__()

        self.block1 = _ConvBlock(1, 32, kernel_size=8, pool_size=8, pool_stride=4, use_elu=True, dropout_p=0.1, bn_momentum=0.01)
        self.block2 = _ConvBlock(32, 64, kernel_size=8, pool_size=8, pool_stride=4, use_elu=False, dropout_p=0.1, bn_momentum=0.01)
        self.block3 = _ConvBlock(64, 128, kernel_size=8, pool_size=8, pool_stride=4, use_elu=False, dropout_p=0.1, bn_momentum=0.01)
        self.block4 = _ConvBlock(128, 256, kernel_size=8, pool_size=8, pool_stride=4, use_elu=False, dropout_p=0.1, bn_momentum=0.01)

        # Compute flattened feature length after 4 pools
        feat_len = self._compute_feature_len(5000, 8, 4, num_pools=4)
        self._feat_len = feat_len * 256

        self.fc1 = nn.Linear(self._feat_len, 512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.01)
        self.drop_fc1 = nn.Dropout(p=0.7)

        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.01)
        self.drop_fc2 = nn.Dropout(p=0.5)

        self.classifier = nn.Linear(512, num_classes)

        self._init_fc()

    @staticmethod
    def _compute_feature_len(L: int, pool_k: int, pool_s: int, *, num_pools: int) -> int:
        out = L
        for _ in range(num_pools):
            out = (out - pool_k) // pool_s + 1
        return out

    def _init_fc(self) -> None:
        for m in [self.fc1, self.fc2, self.classifier]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # BN init
        nn.init.ones_(self.bn_fc1.weight); nn.init.zeros_(self.bn_fc1.bias)
        nn.init.ones_(self.bn_fc2.weight); nn.init.zeros_(self.bn_fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, L) or (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(1) == 1, f"Expected shape (B, 1, L) or (B, L); got {tuple(x.shape)}"

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.flatten(start_dim=1)  # (B, C*L)

        x = F.relu(self.bn_fc1(self.fc1(x)), inplace=True)
        x = self.drop_fc1(x)

        x = F.relu(self.bn_fc2(self.fc2(x)), inplace=True)
        x = self.drop_fc2(x)

        return self.classifier(x)

if __name__ == "__main__":
    torch.manual_seed(0)
    m = DFNoDefNet()

