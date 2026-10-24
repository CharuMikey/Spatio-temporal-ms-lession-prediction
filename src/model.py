
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class TemporalAttention(nn.Module):
    def __init__(self, feature_size):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, sequence_features):
        # Calculate attention scores for each MRI slice
        scores = self.attention(sequence_features)

        # Convert scores into attention weights
        weights = torch.softmax(scores, dim=1)

        # Combine slice features using attention weights
        context = torch.sum(
            weights * sequence_features,
            dim=1
        )

        return context, weights.squeeze(-1)


class SpatioTemporalMSModel(nn.Module):
    def __init__(self, hidden_size=256, num_lstm_layers=1):
        super().__init__()

        # Step 1: EfficientNet-B0 extracts spatial features
        backbone = efficientnet_b0(weights=None)

        self.cnn_features = backbone.features
        self.global_pool = backbone.avgpool

        feature_size = 1280

        # Step 2: BiLSTM learns relationships between MRI slices
        self.bilstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        # Bidirectional LSTM produces 2 * hidden_size features
        temporal_size = hidden_size * 2

        # Step 3: Temporal attention
        self.temporal_attention = TemporalAttention(
            temporal_size
        )

        # Step 4: Binary classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(temporal_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Input shape:
        # (batch_size, time_steps, channels, height, width)

        batch_size, time_steps, channels, height, width = x.shape

        # Prepare each MRI slice for EfficientNet
        x = x.reshape(
            batch_size * time_steps,
            channels,
            height,
            width
        )

        # Extract spatial features
        x = self.cnn_features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Restore the sequence structure for BiLSTM
        x = x.reshape(batch_size, time_steps, -1)

        # Learn temporal relationships between slices
        x, _ = self.bilstm(x)

        # Calculate attention and the sequence representation
        context, attention_weights = self.temporal_attention(x)

        # Generate the prediction logit
        logits = self.classifier(context).squeeze(-1)

        return logits, attention_weights
