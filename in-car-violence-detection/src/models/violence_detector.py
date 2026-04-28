"""
Violence detection model architecture.
CNN (ResNet50) + Bidirectional LSTM with Attention.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on relevant regions."""
    
    def __init__(self, in_channels: int, reduction: int = 4):
        super(SpatialAttention, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            attended: (B, C, H, W)
            weights: (B, 1, H, W)
        """
        weights = self.attention(x)
        attended = x * weights
        return attended, weights


class TemporalAttention(nn.Module):
    """Temporal attention for focusing on important time steps."""
    
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (B, T, hidden_dim*2)
        Returns:
            context: (B, hidden_dim*2)
            weights: (B, T, 1)
        """
        weights = self.attention(lstm_output)
        context = (lstm_output * weights).sum(dim=1)
        return context, weights


class InCarViolenceDetector(nn.Module):
    """
    Violence detection model for in-car scenarios.
    
    Architecture:
    1. ResNet50 CNN backbone with spatial attention
    2. Bidirectional LSTM for temporal modeling
    3. Temporal attention for clip-level classification
    """
    
    def __init__(self, 
                 num_classes: int = 2,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 backbone: str = "resnet50"):
        super(InCarViolenceDetector, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # CNN Backbone
        if backbone == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
            cnn_output_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(cnn_output_dim)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize LSTM and classifier weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, frames, channels, height, width) = (B, 16, 3, 224, 224)
        
        Returns:
            output: (B, num_classes)
            temporal_weights: (B, T, 1)
            spatial_weights: List of (B, 1, H, W)
        """
        batch_size, num_frames = x.shape[:2]
        
        # CNN feature extraction
        cnn_features = []
        spatial_weights_list = []
        
        for t in range(num_frames):
            frame = x[:, t, :, :, :]  # (B, 3, 224, 224)
            feat = self.features(frame)  # (B, 2048, 7, 7)
            
            # Spatial attention
            attended_feat, spatial_w = self.spatial_attention(feat)
            spatial_weights_list.append(spatial_w)
            
            # Global average pooling
            feat_vec = attended_feat.mean(dim=[2, 3])  # (B, 2048)
            cnn_features.append(feat_vec)
        
        # Stack: (B, T, 2048)
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_features)  # (B, T, 1024)
        
        # Temporal attention
        context, temporal_weights = self.temporal_attention(lstm_out)
        
        # Classification
        output = self.classifier(context)
        
        return output, temporal_weights, spatial_weights_list
    
    def predict(self, x):
        """
        Inference mode - returns probabilities and predictions.
        
        Args:
            x: (B, T, C, H, W)
        
        Returns:
            probs: (B, num_classes)
            preds: (B,)
            violence_score: (B,) - probability of violence class
        """
        self.eval()
        with torch.no_grad():
            output, _, _ = self.forward(x)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            violence_score = probs[:, 1] if self.num_classes == 2 else probs.max(dim=1)[0]
        
        return probs, preds, violence_score


def test_model():
    """Quick test of model."""
    print("Testing InCarViolenceDetector...")
    
    model = InCarViolenceDetector(num_classes=2)
    
    # Test input: batch=2, frames=16, channels=3, height=224, width=224
    x = torch.randn(2, 16, 3, 224, 224)
    
    output, temporal_w, spatial_w = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Temporal weights shape: {temporal_w.shape}")
    print(f"Number of spatial weights: {len(spatial_w)}")
    print(f"Spatial weight shape: {spatial_w[0].shape}")
    
    # Test predict
    probs, preds, scores = model.predict(x)
    print(f"Probs shape: {probs.shape}")
    print(f"Preds shape: {preds.shape}")
    print(f"Violence scores: {scores}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("✅ Model test passed!")


if __name__ == "__main__":
    test_model()
