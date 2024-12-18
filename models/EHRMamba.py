import torch
import torch.nn as nn

from .mamba import Mamba as MambaBlock, MambaConfig

class MambaEHR(nn.Module):
    def __init__(self, ts_dim, static_dim, latent_dim=32, d_state=16, n_layers=4, expand_factor=2):
        super().__init__()

        self.mamba_model_dim = 32

        # Encode static features
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, self.mamba_model_dim),
            nn.ReLU(),
            nn.Linear(self.mamba_model_dim, latent_dim)  # Match to latent_dim
        )

        # Linear projection from ts_dim to model_dim
        self.ts_projection = nn.Linear(ts_dim, self.mamba_model_dim)

        # Mamba block for time-series modeling
        self.mamba_block = MambaBlock(
            MambaConfig(
                d_model=self.mamba_model_dim,  # Model dimension d_model
                d_state=d_state,    # SSM state expansion factor
                n_layers=n_layers,      # Local convolution width
                expand_factor=expand_factor       # Block expansion factor
            )
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output mortality prediction as probability
        )

    def forward(self, ts_values, ts_indicators, static_features):
        """
        Forward pass through the model.

        Args:
            ts_values: Tensor of shape (batch_size, seq_len, ts_dim)
            ts_indicators: Tensor of shape (batch_size, seq_len, ts_dim)
            static_features: Tensor of shape (batch_size, static_dim)
        Returns:
            mortality_pred: Tensor of shape (batch_size, 1)
        """

        # Mask time-series values
        masked_values = ts_values * ts_indicators  # Mask out padded regions

        # Project time-series features to model dimension
        ts_proj = self.ts_projection(masked_values)  # (batch_size, seq_len, model_dim)

        # Pass through the Mamba block
        ts_features = self.mamba_block(ts_proj)  # (batch_size, seq_len, model_dim)

        # Aggregate time-series features (e.g., via mean pooling)
        ts_features_agg = ts_features.mean(dim=1)  # (batch_size, model_dim)

        # Encode static features
        static_encoded = self.static_encoder(static_features)  # (batch_size, latent_dim)

        # Combine static and time-series features
        combined_features = static_encoded + ts_features_agg  # (batch_size, latent_dim)

        # Classification head
        mortality_pred = self.classifier(combined_features)  # (batch_size, 1)

        return mortality_pred


# The following models are used within this project's "interface" and trained on its preprocessed data

class OneLayerMambaEHRModel(nn.Module):
    def __init__(
        self, 
        ts_dim, 
        static_dim, 
        hidden_dim, 
        output_dim, 
        seq_len, 
        dropout_rate=0.5  # Moderate dropout for regularization
    ):
        super().__init__()
        
        # Mamba layer for time-series data
        self.mamba_layer = MambaBlock(
            MambaConfig(
                d_model=ts_dim,  # Feature dimension
                n_layers=20,  # Minimal layers for simplicity
                d_state=20,  # Hidden state size
                expand_factor=8,  # Default expansion factor
            )
        )

        # Static encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2 + ts_dim, 32),  # Combine static and Mamba features
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)  # Output layer
        )

    def forward(self, x, static, time, sensor_mask, **kwargs) -> torch.Tensor:
        """
        Forward pass for the simple EHR model with Mamba.
        """
        # Apply mask to time-series data
        x = x * sensor_mask  # Shape: (B, ts_dim, seq_len)
        x = x.transpose(1, 2)  # Shape: (B, seq_len, ts_dim)

        # Pass through Mamba layer
        ts_features = self.mamba_layer(x)  # Shape: (B, seq_len, ts_dim)
        ts_features = torch.mean(ts_features, dim=1)  # Aggregate over time dimension: (B, ts_dim)

        # Encode static data
        static_features = self.static_encoder(static)  # Shape: (B, hidden_dim // 2)

        # Combine features
        combined = torch.cat([ts_features, static_features], dim=1)  # Shape: (B, hidden_dim // 2 + ts_dim)

        # Classify combined features
        output = self.classifier(combined)  # Shape: (B, output_dim)

        return output


class MambaEHRModel(nn.Module):
    def __init__(
        self, 
        ts_dim, 
        static_dim, 
        output_dim, 
        seq_len, 
        dropout_rate=0.5,
        hidden_dim=16,
        mamba_model_dim=16,
        num_mambas_ensemble=8,
    ):
        super().__init__()
        
        self.mamba_model_dim = mamba_model_dim
        self.seq_len = seq_len
        
        # Linear layer for projecting time-series data
        # We will apply BatchNorm after we transpose to (B, C, L) format
        self.ts_projection = nn.Linear(ts_dim, self.mamba_model_dim)
        self.ts_bn = nn.BatchNorm1d(self.mamba_model_dim)  # Normalizes over features (C)

        # Mamba layer for time-series data
        self.mamba_layer = MambaBlock(
            MambaConfig(
                d_model=self.mamba_model_dim,  # D
                n_layers=20,
                d_state=64,
                expand_factor=2,
            )
        )
        
        # Create an ensemble of Mamba layers
        self.mamba_ensemble = nn.ModuleList([
            MambaBlock(
                MambaConfig(
                    d_model=self.mamba_model_dim,  # D
                    n_layers=16,
                    d_state=64,
                    expand_factor=2,
                )
            )
            for _ in range(num_mambas_ensemble)
        ])
                
        self.num_mambas_ensemble = num_mambas_ensemble

        # Static encoder with BatchNorm
        # For static features, shape is (B, static_dim), so BatchNorm1d works on (B, hidden_dim)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Classifier with BatchNorm
        # After concatenation, shape is (B, hidden_dim//2 + mamba_model_dim * num_mambas_ensemble)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2 + self.mamba_model_dim * self.num_mambas_ensemble, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, output_dim)  # Output layer
        )

    def forward(self, x, static, time, sensor_mask, **kwargs) -> torch.Tensor:
        """
        Forward pass for the simple EHR model with Mamba.
        """
        # x: (B, ts_dim, seq_len)
        # Apply mask to time-series data
        x = x * sensor_mask  # Still (B, ts_dim, seq_len)
        
        # Move features to last dimension: (B, seq_len, ts_dim)
        x = x.transpose(1, 2)
        
        # Project to mamba_model_dim: (B, seq_len, mamba_model_dim)
        ts_projection = self.ts_projection(x)
        
        # For BatchNorm1d, we need (B, C, L) format, where C is the feature dimension
        ts_projection = ts_projection.transpose(1, 2)  # (B, mamba_model_dim, seq_len)
        ts_projection = self.ts_bn(ts_projection)       # Normalized over features
        ts_projection = ts_projection.transpose(1, 2)   # Back to (B, seq_len, mamba_model_dim)

        # Pass through Mamba ensemble
        mamba_outputs = []
        for mamba in self.mamba_ensemble:
            ts_features = mamba(ts_projection)        # (B, seq_len, mamba_model_dim)
            ts_features = torch.mean(ts_features, dim=1)  # Aggregate over time -> (B, mamba_model_dim)
            mamba_outputs.append(ts_features)
        
        ts_features_ensemble = torch.cat(mamba_outputs, dim=1)  # (B, mamba_model_dim * num_mambas_ensemble)
        
        # Encode static data: (B, static_dim) -> (B, hidden_dim//2)
        static_features = self.static_encoder(static)

        # Combine features
        combined = torch.cat([ts_features_ensemble, static_features], dim=1)  
        # Shape: (B, hidden_dim//2 + mamba_model_dim * num_mambas_ensemble)
        
        # Classify combined features
        output = self.classifier(combined)  # (B, output_dim)

        return output


class MultiHeadAttentionMambaEHRModel(nn.Module):
    def __init__(
        self, 
        ts_dim, 
        static_dim, 
        hidden_dim, 
        output_dim, 
        seq_len, 
        num_mambas=5, 
        dropout_rate=0.5,  # Increased dropout for regularization
        num_heads=6,  # Attention heads
    ):
        super().__init__()
        
        # Adjust embed_dim for attention compatibility
        self.attention_embed_dim = ts_dim - (ts_dim % num_heads)  # Closest multiple of num_heads <= ts_dim

        # Create an ensemble of Mamba layers
        self.mamba_ensemble = nn.ModuleList([
            MambaBlock(
                MambaConfig(
                    d_model=ts_dim,  # D
                    n_layers=4,  # Reduced number of layers per Mamba
                    d_state=16,  # Hidden state size
                    expand_factor=2,  # Default expand factor
                )
            ) for _ in range(num_mambas)
        ])
                
        self.num_mambas = num_mambas
        self.seq_len = seq_len
        self.ts_dim = ts_dim

        # Attention layer for time-series data
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.attention_embed_dim, num_heads=num_heads, dropout=dropout_rate)

        # Static encoder with a residual connection
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)  # Residual layer
        )

        # Residual connection for static features
        self.residual_layer = nn.Sequential(
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )

        # Improved classifier with higher dropout and smaller intermediate layers
        self.classifier = nn.Sequential(
            nn.Linear(self.attention_embed_dim * num_mambas + hidden_dim // 2, 32),  # Reduced size
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),  # Smaller intermediate layer
            nn.ReLU(),
            nn.Linear(16, output_dim)  # Output layer (no activation, logits expected)
        )

    def forward(self, x, static, time, sensor_mask, **kwargs) -> torch.Tensor:
        """ 
        Forward pass for the Mamba-based EHR model.
        """
        # Batch Size (B): The number of data samples in a batch (e.g., 32).
        # Feature Dimension (ts_dim): The number of sensors or features (e.g., 37). This corresponds to the second dimension in x when it is first passed to forward.
        # Time-Indexed Features (T or variable_dim): The number of time steps, readings, or time-indexed features, which is dynamic.
        # T can vary depending on the preprocessing and masking operations.
        # Apply mask to time-series data
        x = x * sensor_mask  # Shape: (B, ts_dim, T)
        x = x.transpose(1, 2)  # Shape: (B, T, ts_dim)

        # Pass through Mamba ensemble and attention layer
        mamba_outputs = []
        for mamba in self.mamba_ensemble:
            ts_features = mamba(x)
            ts_features = ts_features[..., :self.attention_embed_dim]  # Adjust for attention
            ts_features, _ = self.attention_layer(ts_features, ts_features, ts_features)
            ts_features = torch.mean(ts_features, dim=1)  # Aggregate over time
            mamba_outputs.append(ts_features)
        
        ts_features_ensemble = torch.cat(mamba_outputs, dim=1)  # Shape: (B, attention_embed_dim * num_mambas)

        # Process static data with residual connection
        static_features = self.static_encoder(static)  # Shape: (B, hidden_dim // 2)
        # Match static shape to static_features shape before residual connection
        static_resized = nn.Linear(static.size(1), static_features.size(1)).to(static.device)(static)
        static_features_residual = self.residual_layer(static_features + static_resized)

        # Combine time-series and static features
        combined = torch.cat([ts_features_ensemble, static_features_residual], dim=1)
        output = self.classifier(combined)

        return output
