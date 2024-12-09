import torch
import torch.nn as nn

from .mamba import Mamba as MambaLayer, MambaConfig

class SimpleMambaEHRModel(nn.Module):
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
        self.mamba_layer = MambaLayer(
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


class SimpleMambaEHRModel(nn.Module):
    def __init__(
        self, 
        ts_dim, 
        static_dim, 
        output_dim, 
        seq_len, 
        dropout_rate=0,
        hidden_dim=16, 
        mamba_model_dim=16,
        num_mambas_ensemble=4,
    ):
        super().__init__()
        
        self.mamba_model_dim = mamba_model_dim
        self.seq_len = seq_len
        
        # Linear layer for projecting time-series data
        # We will apply BatchNorm after we transpose to (B, C, L) format
        self.ts_projection = nn.Linear(ts_dim, self.mamba_model_dim)
        self.ts_bn = nn.BatchNorm1d(self.mamba_model_dim)  # Normalizes over features (C)

        # Mamba layer for time-series data
        self.mamba_layer = MambaLayer(
            MambaConfig(
                d_model=self.mamba_model_dim,  # D
                n_layers=2,
                d_state=64,
                expand_factor=2,
            )
        )
        
        # Create an ensemble of Mamba layers
        self.mamba_ensemble = nn.ModuleList([
            MambaLayer(
                MambaConfig(
                    d_model=self.mamba_model_dim,  # D
                    n_layers=4,
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


class MambaEHRModel(nn.Module):
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
            MambaLayer(
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
    


# class MambaLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(MambaLayer, self).__init__()
#         self.hidden_dim = hidden_dim

#         # State-space components
#         self.state_transition = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
#         self.input_mapping = nn.Linear(input_dim, hidden_dim)
#         self.output_mapping = nn.Linear(hidden_dim, input_dim)

#         # Input gating for dynamic state adjustments
#         self.input_gate = nn.Linear(input_dim, hidden_dim)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # x: (batch_size, ts_dim, seq_len)
#         batch_size, ts_dim, seq_len = x.size()

#         # Initialize states: (batch_size, ts_dim, hidden_dim)
#         states = torch.zeros(batch_size, ts_dim, self.hidden_dim).to(x.device)
#         outputs = []

#         for t in range(seq_len):
#             input_t = x[:, :, t]  # Shape: (batch_size, ts_dim)
#             print(f"input_t shape: {input_t.shape}")

#             # Gate control for each feature
#             gate_t = self.sigmoid(self.input_gate(input_t))  # (batch_size, hidden_dim)
#             print(f"gate_t shape: {gate_t.shape}")

#             # State transition for each feature
#             for i in range(ts_dim):
#                 states[:, i, :] = (
#                     (1 - gate_t[:, i].unsqueeze(-1)) * states[:, i, :]
#                     + gate_t[:, i].unsqueeze(-1)
#                     * torch.matmul(states[:, i, :], self.state_transition)
#                 )

#             # Map input to state space for each feature
#             mapped_input = self.input_mapping(input_t)  # Shape: (batch_size, ts_dim, hidden_dim)
#             states += mapped_input.unsqueeze(1)  # Expand mapped_input to match (batch_size, ts_dim, hidden_dim)
#             print(f"states shape: {states.shape}")

#             # Map state to output
#             output_t = self.output_mapping(states)  # Shape: (batch_size, ts_dim, input_dim)
#             print(f"output_t shape after mapping: {output_t.shape}")

#             outputs.append(output_t)

#         # Final stacked output: (batch_size, input_dim, seq_len)
#         return torch.stack(outputs, dim=-1)

# class MambaEHRModel(nn.Module):
#     def __init__(self, ts_dim, static_dim, hidden_dim, output_dim, seq_len):
#         super(MambaEHRModel, self).__init__()
#         self.mamba_layer = MambaLayer(ts_dim, hidden_dim)
#         self.static_encoder = nn.Sequential(
#             nn.Linear(static_dim, hidden_dim),
#             nn.ReLU()
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x, static, time, sensor_mask, **kwargs) -> torch.Tensor:
#         # Apply mask to time-series data to handle missing values
#         print(x.shape)
#         x = x * sensor_mask  # Mask time-series data (batch size, number of time-series features = 37, sequence length)
#         print(x.shape)
        
#         # The Mamba layer processes the sequence dimension (151) for each feature dimension (37) independently.
# 	    # The hidden_dim specifies the size of the latent representation for each feature after passing through the state-space model.
#         ts_features = self.mamba_layer(x)
#         print(ts_features.shape)
        
#         raise NotImplementedError
#         # # Process time-series data with Mamba
#         # ts_features = self.mamba_layer(x)  # Shape: (batch_size, ts_dim, seq_len)
#         # ts_features = torch.mean(ts_features, dim=(1, 2))  # Collapse both ts_dim and seq_len
#         # print(f"ts_features shape after full aggregation: {ts_features.shape}")  # Should be (batch_size, hidden_dim)

#         # # Process static data
#         # static_features = self.static_encoder(static)  # Shape: (batch_size, hidden_dim)
#         # print(f"static_features shape: {static_features.shape}")

#         # # Combine time-series and static features
#         # combined = torch.cat([ts_features, static_features], dim=1)  # Shape: (batch_size, hidden_dim * 2)

#         # Classify the combined features
#         return self.classifier(x)


