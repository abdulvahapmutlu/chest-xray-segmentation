import torch
import torch.nn as nn
import timm

class SimpleDecoder(nn.Module):
    """
    A simpler UNet-like decoder without attention blocks.
    Uses skip connections, transposed convolutions, and a final upsample.
    """
    def __init__(self, encoder_channels, out_channels=1):
        super(SimpleDecoder, self).__init__()
        self.conv_f4 = nn.Conv2d(encoder_channels[3], 512, kernel_size=1)
        self.conv_f3 = nn.Conv2d(encoder_channels[2], 256, kernel_size=1)
        self.conv_f2 = nn.Conv2d(encoder_channels[1], 128, kernel_size=1)
        self.conv_f1 = nn.Conv2d(encoder_channels[0], 64, kernel_size=1)
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        
    def forward(self, features):
        # features: [f1, f2, f3, f4]
        f1, f2, f3, f4 = features
        f4 = self.conv_f4(f4)
        f3 = self.conv_f3(f3)
        f2 = self.conv_f2(f2)
        f1 = self.conv_f1(f1)
        
        x = f4
        x = self.up1(x)
        x = self.fuse1(torch.cat([x, f3], dim=1))
        x = self.up2(x)
        x = self.fuse2(torch.cat([x, f2], dim=1))
        x = self.up3(x)
        x = self.fuse3(torch.cat([x, f1], dim=1))
        x = self.up4(x)
        x = self.up5(x)
        x = self.out_conv(x)
        return x

class SwinTransformerSegModel(nn.Module):
    """
    Full segmentation model with a Swin Transformer encoder and custom decoder.
    """
    def __init__(self, backbone_name="swin_tiny_patch4_window7_224", out_channels=1):
        super(SwinTransformerSegModel, self).__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=True, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()
        self.decoder = SimpleDecoder(encoder_channels, out_channels)
    
    def forward(self, x):
        features = self.encoder(x)
        # Ensure features are in channels-first format if needed.
        permuted_features = []
        for f in features:
            if f.dim() == 4 and f.shape[1] < f.shape[-1]:
                f = f.permute(0, 3, 1, 2)
            permuted_features.append(f)
        seg_map = self.decoder(permuted_features)
        return seg_map
