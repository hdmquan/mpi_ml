import torch
import torch.nn as nn

from src.utils import set_seed

set_seed()


class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # No. Fourier modes in 1st dim
        self.modes2 = modes2  # No. Fourier modes in 2nd dim

        self.scale = 1 / (in_channels * out_channels)

        # Random init with complex (cfloat) weights
        # Note to future self: FFT is "3D"
        self.weights1 = nn.Parameter(
            self.scale
            * torch.randn(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

        self.weights2 = nn.Parameter(
            self.scale
            * torch.randn(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # rfft2 is the 2D FFT for real-valued signals
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Lower freqs spectral conv
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )

        # Optionnel. Higher freqs spectral conv
        if self.modes1 < x.size(-2) // 2 + 1:
            out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, -self.modes1 :, : self.modes2],
                self.weights2,
            )

        # Un-fft the thing
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock2D(nn.Module):

    def __init__(self, channels, modes1, modes2):
        super().__init__()

        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.bn(self.spectral_conv(x) + self.conv(x)))


class FNOModel2D(nn.Module):

    def __init__(
        self,
        in_channels=7,
        out_channels=6,
        modes1=12,
        modes2=12,
        width=32,
        num_layers=2,
    ):
        super().__init__()

        self.fc_in = nn.Conv2d(in_channels, width, kernel_size=1)

        self.fno_blocks = nn.ModuleList(
            [FNOBlock2D(width, modes1, modes2) for _ in range(num_layers)]
        )

        self.fc_out = nn.Conv2d(width, out_channels, kernel_size=1)
        self.activation = nn.Softplus()

    def forward(self, x):

        x = self.fc_in(x)

        for block in self.fno_blocks:
            x = block(x)

        return self.activation(self.fc_out(x))

    @staticmethod
    def get_grid(shape, device):
        """Generate normalized coordinate grid [-1,1] x [-1,1]"""
        height, width = shape

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Modes in z-direction (altitude)
        self.modes2 = modes2  # Modes in y-direction (latitude)
        self.modes3 = modes3  # Modes in x-direction (longitude)

        self.scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            self.scale
            * torch.randn(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

        self.weights2 = nn.Parameter(
            self.scale
            * torch.randn(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # 3D FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Lower frequencies
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3],
            self.weights1,
        )

        # Higher frequencies (optional)
        if self.modes1 < x.size(-3) // 2 + 1:
            out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = torch.einsum(
                "bixyz,ioxyz->boxyz",
                x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3],
                self.weights2,
            )

        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


class FNOBlock3D(nn.Module):
    def __init__(self, channels, modes1, modes2, modes3):
        super().__init__()

        self.spectral_conv = SpectralConv3d(channels, channels, modes1, modes2, modes3)
        self.conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.bn(self.spectral_conv(x) + self.conv(x)))


class FNOModel3D(nn.Module):
    def __init__(
        self,
        in_channels=5,
        out_channels=6,
        modes1=8,  # altitude modes
        modes2=12,  # latitude modes
        modes3=12,  # longitude modes
        width=32,
        num_layers=2,
    ):
        super().__init__()

        self.fc_in = nn.Conv3d(in_channels, width, kernel_size=1)

        self.fno_blocks = nn.ModuleList(
            [FNOBlock3D(width, modes1, modes2, modes3) for _ in range(num_layers)]
        )

        self.fc_out = nn.Conv3d(width, out_channels, kernel_size=1)
        self.activation = nn.Softplus()

    def forward(self, x):
        x = self.fc_in(x)

        for block in self.fno_blocks:
            x = block(x)

        return self.activation(self.fc_out(x))

    @staticmethod
    def get_grid(shape, device):
        """Generate normalized coordinate grid [-1,1] x [-1,1] x [-1,1]"""
        depth, height, width = shape

        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, depth, device=device),
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        return torch.stack([grid_x, grid_y, grid_z], dim=0).unsqueeze(0)
