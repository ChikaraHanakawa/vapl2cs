import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Optional, Dict, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism
    """
    def __init__(self, scale, dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, attn_mask=None):
        """
        Args:
            q: Query of shape (B, ..., L, d_k)
            k: Key of shape (B, ..., L, d_k)
            v: Value of shape (B, ..., L, d_v)
            attn_mask: Optional mask of shape (B, ..., L, L)
            
        Returns:
            Attention output and attention weights
        """
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v)
        
        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(scale=self.d_k ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # LayerNorm for the feature dimension only
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, q, k, v, attn_mask=None):
        """
        Args:
            q: Query of shape (B, L_q, d_model) or (B, d_model, L_q)
            k: Key of shape (B, L_k, d_model) or (B, d_model, L_k)
            v: Value of shape (B, L_v, d_model) or (B, d_model, L_v)
            attn_mask: Optional mask of shape (B, L_q, L_k)
            
        Returns:
            Attention output and attention weights
        """
        batch_size = q.size(0)
        residual = q
        
        # Check if tensors are in format [B, D, L] and transpose if needed
        q_format_transposed = len(q.shape) == 3 and q.size(1) == self.d_model and q.size(2) != self.d_model
        k_format_transposed = len(k.shape) == 3 and k.size(1) == self.d_model and k.size(2) != self.d_model
        v_format_transposed = len(v.shape) == 3 and v.size(1) == self.d_model and v.size(2) != self.d_model
        
        # Remember original format to return in same format
        original_format_transposed = q_format_transposed
        
        # Transpose to [B, L, D] format if needed
        if q_format_transposed:
            q = q.transpose(1, 2)
        if k_format_transposed:
            k = k.transpose(1, 2)
        if v_format_transposed:
            v = v.transpose(1, 2)
        
        # Linear projection and reshape
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Reshape attention mask if provided
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # Apply attention
        output, attn_weights = self.attention(q, k, v, attn_mask)
        
        # Reshape and concat
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(output)
        output = self.dropout(output)
        
        # For residual connection, make sure format matches
        if original_format_transposed:
            residual = residual.transpose(1, 2)
        
        # Add residual connection
        output = residual + output
        
        # Apply layer norm
        output = self.layer_norm(output)
        
        # Return in the original format
        if original_format_transposed:
            output = output.transpose(1, 2)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # LayerNorm for the feature dimension only
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        
    def forward(self, x):
        residual = x
        
        # Check if the tensor is in [B, D, L] format and handle accordingly
        is_transposed = len(x.shape) == 3 and x.size(1) == self.d_model and x.size(2) != self.d_model
        
        if is_transposed:
            # Transpose to [B, L, D] for processing with linear layers
            x_t = x.transpose(1, 2)
            
            # Apply feedforward layers
            x_t = F.relu(self.fc1(x_t))
            x_t = self.dropout(x_t)
            x_t = self.fc2(x_t)
            x_t = self.dropout(x_t)
            
            # Add residual connection (need to transpose residual)
            residual_t = residual.transpose(1, 2)
            x_t = residual_t + x_t
            
            # Apply layer norm
            x_t = self.layer_norm(x_t)
            
            # Transpose back to [B, D, L]
            return x_t.transpose(1, 2)
        else:
            # Standard [B, L, D] processing
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            
            # Add residual connection
            x = residual + x
            
            # Apply layer norm
            x = self.layer_norm(x)
            
            return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the Transformer model
    """
    def __init__(self, d_model, max_seq_len=1000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        self.d_model = d_model
        
    def forward(self, x):
        """
        Args:
            x: Input tensor that can be of shape (B, L, d_model) or (B, d_model, L)
        """
        # Check if input is in format [B, d_model, L]
        if x.size(1) == self.d_model and len(x.shape) == 3:
            # Need to apply positional encoding in transposed space
            x_transposed = x.transpose(1, 2)  # [B, L, d_model]
            x_with_pe = x_transposed + self.pe[:, :x_transposed.size(1)]
            return x_with_pe.transpose(1, 2)  # Return to [B, d_model, L]
        else:
            # Standard [B, L, d_model] format
            return x + self.pe[:, :x.size(1)]


class MultimodalTransformerBlock(nn.Module):
    """
    Multimodal Transformer Block combining audio and visual features
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Self-attention layers for individual modalities
        self.self_attn_a1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.self_attn_a2 = MultiHeadAttention(d_model, num_heads, dropout)
        self.self_attn_v1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.self_attn_v2 = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Cross-attention layers between audio and audio
        self.cross_attn_a1_a2 = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn_a2_a1 = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Cross-attention layers between audio and visual (same speaker)
        self.cross_attn_a1_v1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn_a2_v2 = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Cross-attention layers between visual and audio (same speaker)
        self.cross_attn_v1_a1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn_v2_a2 = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed forward layers
        self.ff_a1 = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ff_a2 = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ff_v1 = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ff_v2 = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Fusion layers
        self.fusion_a1 = nn.Linear(d_model * 3, d_model)
        self.fusion_a2 = nn.Linear(d_model * 3, d_model)
        self.fusion_v1 = nn.Linear(d_model * 2, d_model)
        self.fusion_v2 = nn.Linear(d_model * 2, d_model)
        
        self.dropout = nn.Dropout(dropout)
        # LayerNorm for the feature dimension only
        self.layer_norm_a1 = nn.LayerNorm(d_model)
        self.layer_norm_a2 = nn.LayerNorm(d_model)
        self.layer_norm_v1 = nn.LayerNorm(d_model)
        self.layer_norm_v2 = nn.LayerNorm(d_model)
        
    def forward(self, a1, a2, v1=None, v2=None, attn_mask=None):
        """
        Forward pass through the multimodal transformer block
        
        Args:
            a1: Audio features for speaker 1 (B, L, d_model) or (B, d_model, L)
            a2: Audio features for speaker 2 (B, L, d_model) or (B, d_model, L)
            v1: Visual features for speaker 1 (B, L, d_model), (B, d_model, L) or None
            v2: Visual features for speaker 2 (B, L, d_model), (B, d_model, L) or None
            attn_mask: Optional attention mask
            
        Returns:
            Updated feature representations and attention weights
        """
        # Process audio features
        # Self-attention
        a1_self, a1_self_attn = self.self_attn_a1(a1, a1, a1, attn_mask)
        a2_self, a2_self_attn = self.self_attn_a2(a2, a2, a2, attn_mask)
        
        # Cross-attention between audio streams
        a1_cross_a2, a1_cross_a2_attn = self.cross_attn_a1_a2(a1, a2, a2, attn_mask)
        a2_cross_a1, a2_cross_a1_attn = self.cross_attn_a2_a1(a2, a1, a1, attn_mask)
        
        # Initialize attention outputs for visual features
        a1_cross_v1 = None
        a2_cross_v2 = None
        v1_cross_a1 = None
        v2_cross_a2 = None
        v1_self = None
        v2_self = None
        
        # Process visual features if available
        if v1 is not None and v2 is not None:
            # Self-attention for visual features
            v1_self, v1_self_attn = self.self_attn_v1(v1, v1, v1, attn_mask)
            v2_self, v2_self_attn = self.self_attn_v2(v2, v2, v2, attn_mask)
            
            # Cross-attention between audio and visual (same speaker)
            a1_cross_v1, a1_cross_v1_attn = self.cross_attn_a1_v1(a1, v1, v1, attn_mask)
            a2_cross_v2, a2_cross_v2_attn = self.cross_attn_a2_v2(a2, v2, v2, attn_mask)
            
            # Cross-attention between visual and audio (same speaker)
            v1_cross_a1, v1_cross_a1_attn = self.cross_attn_v1_a1(v1, a1, a1, attn_mask)
            v2_cross_a2, v2_cross_a2_attn = self.cross_attn_v2_a2(v2, a2, a2, attn_mask)
            
            # Fusion for visual features
            
            # 全ての処理を完全に明確化
            # まずv1_selfとv1_cross_a1を[B, L, D]形式に揃える
            if len(v1_self.shape) == 3 and v1_self.size(1) == self.layer_norm_v1.normalized_shape[0]:
                # [B, D, L] -> [B, L, D]に転置
                v1_self_t = v1_self.transpose(1, 2)
                v1_cross_a1_t = v1_cross_a1.transpose(1, 2)
            else:
                # 既に[B, L, D]形式の場合
                v1_self_t = v1_self
                v1_cross_a1_t = v1_cross_a1
            
            # [B, L, D]形式で連結（最後の次元で）
            v1_fusion = torch.cat([v1_self_t, v1_cross_a1_t], dim=-1)  # [B, L, 2D]
            
            v1_fusion = self.fusion_v1(v1_fusion)  # これで [B, L, D] になる
            v1_fusion = self.dropout(v1_fusion)
            
            # 完全に正規化の処理方法を再設計
            # v1とv1_fusionは異なる形式になっている可能性がある
            
            # まず両方のテンソルを[B, L, D]形式に揃える
            if len(v1.shape) == 3 and v1.size(1) == self.layer_norm_v1.normalized_shape[0]:
                # v1が[B, D, L]形式の場合は転置
                v1_t = v1.transpose(1, 2)  # [B, D, L] -> [B, L, D]
            else:
                # v1が既に[B, L, D]形式の場合
                v1_t = v1
                
            # v1_fusionが[B, L, D]形式でない場合は転置
            if len(v1_fusion.shape) == 3 and v1_fusion.size(2) != self.layer_norm_v1.normalized_shape[0]:
                # [B, D, L]形式の場合は転置
                v1_fusion_t = v1_fusion.transpose(1, 2)  # [B, D, L] -> [B, L, D]
            else:
                # 既に[B, L, D]形式の場合
                v1_fusion_t = v1_fusion
                
            # 残差接続（両方とも[B, L, D]形式になっている）
            residual_sum_t = v1_t + v1_fusion_t  # [B, L, D]
            
            # LayerNorm適用（LayerNormは[B, L, D]形式のテンソルに対して最後の次元に適用）
            normalized_t = self.layer_norm_v1(residual_sum_t)  # [B, L, D]
            
            # 元のv1の形式に合わせて戻す
            if len(v1.shape) == 3 and v1.size(1) == self.layer_norm_v1.normalized_shape[0]:
                # v1が[B, D, L]形式だった場合、結果も[B, D, L]に戻す
                v1_out = normalized_t.transpose(1, 2)  # [B, L, D] -> [B, D, L]
            else:
                # v1が[B, L, D]形式だった場合
                v1_out = normalized_t
            
            # Feed-Forwardネットワークを適用
            v1_out = self.ff_v1(v1_out)
            
            # v2に対しても同様の完全に明確化した処理
            # まずv2_selfとv2_cross_a2を[B, L, D]形式に揃える
            if len(v2_self.shape) == 3 and v2_self.size(1) == self.layer_norm_v2.normalized_shape[0]:
                # [B, D, L] -> [B, L, D]に転置
                v2_self_t = v2_self.transpose(1, 2)
                v2_cross_a2_t = v2_cross_a2.transpose(1, 2)
            else:
                # 既に[B, L, D]形式の場合
                v2_self_t = v2_self
                v2_cross_a2_t = v2_cross_a2
            
            # [B, L, D]形式で連結（最後の次元で）
            v2_fusion = torch.cat([v2_self_t, v2_cross_a2_t], dim=-1)  # [B, L, 2D]
            
            v2_fusion = self.fusion_v2(v2_fusion)
            v2_fusion = self.dropout(v2_fusion)
            
            # v2も同様に完全に再設計された処理を適用
            # まず両方のテンソルを[B, L, D]形式に揃える
            if len(v2.shape) == 3 and v2.size(1) == self.layer_norm_v2.normalized_shape[0]:
                # v2が[B, D, L]形式の場合は転置
                v2_t = v2.transpose(1, 2)  # [B, D, L] -> [B, L, D]
            else:
                # v2が既に[B, L, D]形式の場合
                v2_t = v2
                
            # v2_fusionが[B, L, D]形式でない場合は転置
            if len(v2_fusion.shape) == 3 and v2_fusion.size(2) != self.layer_norm_v2.normalized_shape[0]:
                # [B, D, L]形式の場合は転置
                v2_fusion_t = v2_fusion.transpose(1, 2)  # [B, D, L] -> [B, L, D]
            else:
                # 既に[B, L, D]形式の場合
                v2_fusion_t = v2_fusion
                
            # 残差接続（両方とも[B, L, D]形式になっている）
            residual_sum_t = v2_t + v2_fusion_t  # [B, L, D]
            
            # LayerNorm適用
            normalized_t = self.layer_norm_v2(residual_sum_t)  # [B, L, D]
            
            # 元のv2の形式に合わせて戻す
            if len(v2.shape) == 3 and v2.size(1) == self.layer_norm_v2.normalized_shape[0]:
                # v2が[B, D, L]形式だった場合、結果も[B, D, L]に戻す
                v2_out = normalized_t.transpose(1, 2)  # [B, L, D] -> [B, D, L]
            else:
                # v2が[B, L, D]形式だった場合
                v2_out = normalized_t
            
            # Feed-Forwardネットワークを適用
            v2_out = self.ff_v2(v2_out)
        else:
            v1_out = None
            v2_out = None
        
        # Fusion for audio features
        if a1_cross_v1 is not None:
            a1_fusion = torch.cat([a1_self, a1_cross_a2, a1_cross_v1], dim=-1)
        else:
            a1_fusion = torch.cat([a1_self, a1_cross_a2, a1_self], dim=-1)  # Use self attn as placeholder
            
        a1_fusion = self.fusion_a1(a1_fusion)
        a1_fusion = self.dropout(a1_fusion)
        a1_out = self.layer_norm_a1(a1 + a1_fusion)
        a1_out = self.ff_a1(a1_out)
        
        if a2_cross_v2 is not None:
            a2_fusion = torch.cat([a2_self, a2_cross_a1, a2_cross_v2], dim=-1)
        else:
            a2_fusion = torch.cat([a2_self, a2_cross_a1, a2_self], dim=-1)  # Use self attn as placeholder
            
        a2_fusion = self.fusion_a2(a2_fusion)
        a2_fusion = self.dropout(a2_fusion)
        a2_out = self.layer_norm_a2(a2 + a2_fusion)
        a2_out = self.ff_a2(a2_out)
        
        # Collect attention weights
        attn_weights = {
            'a1_self': a1_self_attn,
            'a2_self': a2_self_attn,
            'a1_cross_a2': a1_cross_a2_attn,
            'a2_cross_a1': a2_cross_a1_attn,
        }
        
        if v1 is not None and v2 is not None:
            attn_weights.update({
                'v1_self': v1_self_attn,
                'v2_self': v2_self_attn,
                'a1_cross_v1': a1_cross_v1_attn,
                'a2_cross_v2': a2_cross_v2_attn,
                'v1_cross_a1': v1_cross_a1_attn,
                'v2_cross_a2': v2_cross_a2_attn,
            })
        
        return a1_out, a2_out, v1_out, v2_out, attn_weights


class MultimodalTransformer(nn.Module):
    """
    Multimodal Transformer for integrating audio and visual features
    """
    def __init__(
        self,
        dim=256,
        num_layers=3,
        num_heads=4,
        dff_k=4,
        dropout=0.1,
        max_seq_len=1000,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(dim, max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MultimodalTransformerBlock(
                d_model=dim,
                num_heads=num_heads,
                d_ff=dim * dff_k,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Integration layer
        self.integration = nn.Linear(dim * 2, dim)
        # LayerNorm for the feature dimension only
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, a1, a2, v1=None, v2=None, attention=False):
        """
        Forward pass through the multimodal transformer
        
        Args:
            a1: Audio features for speaker 1 (B, D, L) - 変換されて (B, L, D) で処理
            a2: Audio features for speaker 2 (B, D, L) - 変換されて (B, L, D) で処理
            v1: Visual features for speaker 1 (B, D, L) - 変換されて (B, L, D) で処理
            v2: Visual features for speaker 2 (B, D, L) - 変換されて (B, L, D) で処理
            attention: Whether to return attention weights
            
        Returns:
            Dictionary with integrated features and attention weights if requested
            注意: 出力は (B, L, D) 形式です - 下流のMultimodalVAPモジュールの期待に合わせています
        """
        # Transformerブロックでの処理のために[B, L, D]形式に変換
        # Audio features: [B, D, L] -> [B, L, D]
        if len(a1.shape) == 3 and a1.size(1) == self.dim:
            a1 = a1.transpose(1, 2)
            a2 = a2.transpose(1, 2)
        
        # Visual features（存在する場合）も同様に処理
        if v1 is not None and v2 is not None:
            if len(v1.shape) == 3 and v1.size(1) == self.dim:
                # [B, D, L] -> [B, L, D]
                v1 = v1.transpose(1, 2)
                v2 = v2.transpose(1, 2)
        
        # Apply positional encoding
        
        # Apply positional encoding (pos_encodingは[B, L, D]形式を想定)
        a1 = self.pos_encoding(a1)
        a2 = self.pos_encoding(a2)
        
        if v1 is not None and v2 is not None:
            v1 = self.pos_encoding(v1)
            v2 = self.pos_encoding(v2)
        
        # Initialize attention weights dict if needed
        attn_weights_list = [] if attention else None
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            a1, a2, v1, v2, attn_weights = layer(a1, a2, v1, v2)
            
            if attention:
                attn_weights_list.append(attn_weights)
        
        # 確実に両方のテンソルが[B, L, D]形式であることを確認
        if len(a1.shape) != 3 or a1.size(2) != self.dim:
            if len(a1.shape) == 3 and a1.size(1) == self.dim:
                # [B, D, L] -> [B, L, D]
                a1 = a1.transpose(1, 2)
                a2 = a2.transpose(1, 2)
        
        # Integrate features from both speakers
        
        # [B, L, D]形式で連結して処理
        x = torch.cat([a1, a2], dim=-1)  # [B, L, 2D]
        
        # 線形層を適用 ([B, L, 2D] -> [B, L, D])
        x = self.integration(x)
        x = self.dropout(x)
        
        # Layer Normを適用
        x = self.layer_norm(x)
        
        # エラーメッセージに基づいて下流モジュールの期待形状に合わせる
        # 現在：mat1 and mat2 shapes cannot be multiplied (1024x1000 and 256x1)
        # ここでは、形状を(B, L, D)のままにしておく（転置しない）
        # MultimodalVAP.pyの期待する形状に合わせる
        
        # Prepare output dictionary
        output = {"x": x, "x1": a1, "x2": a2}
        
        if v1 is not None and v2 is not None:
            output["v1"] = v1
            output["v2"] = v2
        
        if attention:
            output["attention"] = attn_weights_list
            
        return output
