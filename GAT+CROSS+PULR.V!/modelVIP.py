import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dropout_edge


# ==============================================================================
# MODULE 2.1: PROJECTION LAYER
# Phóng to 14 chiều -> 64 chiều để tránh Feature Domination
# ==============================================================================
class ProjectionLayer(nn.Module):
    def __init__(self, in_dim=14, out_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU()
        )

    def forward(self, x):
        return self.proj(x)


# ==============================================================================
# MODULE 2.2: SIAMESE DEEP GAT BLOCK
# 1 Block = GATv2Conv + Residual + LayerNorm + ELU
# ==============================================================================
class DeepGATBlock(nn.Module):
    def __init__(self, hidden_dim=64, heads=4, dropout=0.3, drop_edge_p=0.2):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim phải chia hết cho heads"
        self.drop_edge_p = drop_edge_p
        self.dropout = dropout

        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            concat=True,         # Output: hidden_dim // heads * heads = hidden_dim
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.ELU()

    def forward(self, x, edge_index):
        # DropEdge: Loại bỏ ngẫu nhiên 20% cạnh khi train để khử nhiễu topology
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=self.drop_edge_p, training=True)

        out = self.conv(x, edge_index)           # [N, hidden_dim]
        out = self.norm(out + x)                 # Residual Connection
        out = self.act(out)
        return out


# ==============================================================================
# MODULE 2.3: CROSS-ATTENTION + RESIDUAL HINT
# Query=Z_n, Key=Z_t, Value=Z_t; Z_diff = (Z_n - Z_t)^2 làm "hint" tránh Shortcut
# ==============================================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # Nếu dùng concat(Z_attn, Z_diff) -> dim = 128, chiếu về 64
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_n, z_t):
        # MHA yêu cầu input (batch, seq, embed) -> unsqueeze(0) thêm chiều batch ảo
        z_n_b = z_n.unsqueeze(0)   # [1, N, 64]
        z_t_b = z_t.unsqueeze(0)   # [1, N, 64]

        # Cross-attention: Query từ Normal, Key/Value từ Tumor
        z_attn, _ = self.attn(query=z_n_b, key=z_t_b, value=z_t_b)
        z_attn = z_attn.squeeze(0)  # [N, 64]

        # Residual Hint: Bình phương khoảng cách trong không gian tiềm ẩn
        z_diff = (z_n - z_t) ** 2  # [N, 64]

        # Concat rồi chiếu về 64 để giữ nguyên dimension
        z_fusion = torch.cat([z_attn, z_diff], dim=-1)  # [N, 128]
        z_fusion = self.fusion_proj(z_fusion)            # [N, 64]
        z_fusion = self.norm(z_fusion)

        return z_fusion


# ==============================================================================
# MÔ HÌNH CHÍNH: SiamesePUGNN
# ==============================================================================
class SiamesePUGNN(nn.Module):
    """
    Kiến trúc Siamese GAT Cross-Attention cho PU Learning phát hiện gen ung thư.

    Pipeline:
        x_n, x_t  ->  Projection(14->64)
                   ->  Siamese DeepGAT x3 (shared weights)  ->  Z_n, Z_t
                   ->  CrossAttention + Residual Hint        ->  Z_fusion
                   ->  Classifier                            ->  logits
    """
    def __init__(self, in_dim=14, hidden_dim=64, heads=4, num_gat_layers=3,
                 dropout=0.3, drop_edge_p=0.2):
        super().__init__()
        self.dropout_p = dropout

        # Module 2.1 - Shared Projection (cả x_n và x_t dùng chung 1 lớp này)
        self.projection = ProjectionLayer(in_dim=in_dim, out_dim=hidden_dim)

        # Module 2.2 - Siamese Deep GAT (shared weights)
        self.gat_blocks = nn.ModuleList([
            DeepGATBlock(
                hidden_dim=hidden_dim,
                heads=heads,
                dropout=dropout,
                drop_edge_p=drop_edge_p
            )
            for _ in range(num_gat_layers)
        ])

        # Module 2.3 - Cross-Attention Fusion
        self.cross_attn = CrossAttentionFusion(embed_dim=hidden_dim, num_heads=heads)

        # Module 2.4 - Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
            # KHÔNG có Sigmoid -> BCEWithLogitsLoss trong nnPU tự xử lý
        )

    def encode(self, x, edge_index):
        """Mã hóa 1 nhánh qua Projection + 3 DeepGATBlock."""
        h = self.projection(x)
        for block in self.gat_blocks:
            h = block(h, edge_index)
        return h

    def forward(self, x_n, x_t, e_n, e_t=None):
        """
        Args:
            x_n: Tensor [N, in_dim] - đặc trưng gen trong mô bình thường
            x_t: Tensor [N, in_dim] - đặc trưng gen trong mô khối u
            e_n: Tensor [2, E_n]    - edge index đồ thị Normal
            e_t: Tensor [2, E_t]    - edge index đồ thị Tumor (mặc định = e_n nếu None)
        Returns:
            logits: Tensor [N] - raw scores trước sigmoid
        """
        if e_t is None:
            e_t = e_n  # Fallback: dùng chung edge_index đến khi có multiplex graph

        # --- Siamese Encoding (shared weights) ---
        z_n = self.encode(x_n, e_n)   # [N, 64]
        z_t = self.encode(x_t, e_t)   # [N, 64]

        # --- Cross-Attention Fusion ---
        z_fusion = self.cross_attn(z_n, z_t)  # [N, 64]

        # --- Classification ---
        logits = self.classifier(z_fusion).squeeze(-1)  # [N]
        return logits


# ==============================================================================
# HÀM LOSS: ROBUST nnPU LOSS
# Khắc phục: NaN, gradient bùng nổ, prior cố định
# ==============================================================================
def robust_nnpu_loss(logits, labels, prior=0.05):
    """
    Non-negative PU Loss với các cải tiến chống bất ổn số học.

    Args:
        logits : Tensor [N] - raw logits (chưa qua sigmoid)
        labels : Tensor [N] - 1.0=Positive (đã biết), 0.0=Unlabeled
        prior  : float      - Ước lượng tỷ lệ gen ung thư thực trong toàn bộ gen (fix=0.05)

    Returns:
        objective: scalar loss
    """
    pos_mask = (labels == 1.0).float()
    unl_mask = (labels == 0.0).float()

    num_pos = max(pos_mask.sum().item(), 1.0)
    num_unl = max(unl_mask.sum().item(), 1.0)

    # --- Tính các thành phần BCE ---
    # Loss khi nhãn là 1 (gen này là Positive)
    loss_p  = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits),  reduction='none')
    # Loss khi nhãn là 0 (gen này là Negative)
    loss_n  = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')

    # Risk ước tính cho tập Positive
    term_pos     = (loss_p * pos_mask).sum() / num_pos   # E_P[l(f(x), +1)]
    # Risk ước tính cho tập Unlabeled (phía Negative)
    term_unl_neg = (loss_n * unl_mask).sum() / num_unl   # E_U[l(f(x), -1)]
    # Phần "correction" từ Positive (cũng nhãn 0)
    term_pos_neg = (loss_n * pos_mask).sum() / num_pos   # E_P[l(f(x), -1)]

    # --- Non-negative Risk cho Negative class ---
    risk_neg = term_unl_neg - prior * term_pos_neg

    # Kích hoạt Non-negative: nếu âm -> lật dấu để tránh gradient chạy ngược
    if risk_neg.item() < -1e-4:
        # Gradient vẫn truyền qua risk_neg nhưng chiều dương
        objective = prior * term_pos + (-risk_neg)
    else:
        objective = prior * term_pos + risk_neg

    return objective
