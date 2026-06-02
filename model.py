"""
model.py  —  SiamesePUGNN v2
==============================
Nâng cấp so với phiên bản trước:
  [M1] Edge-weighted GATv2       : DeepGATBlock nhận edge_attr (edge_weight)
  [M2] Jumping Knowledge (JK-Net): Lưu output 3 lớp GAT, concat trước fusion
  [M3] Sparse Cross-Attention    : Thay nn.MultiheadAttention O(N²) bằng
                                   MessagePassing graph-aware, chỉ attend
                                   trên các cạnh có trong edge_index
  [M4] robust_nnpu_loss          : Kế thừa nguyên vẹn, chống NaN / gradient flip
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn     import GATv2Conv
from torch_geometric.nn     import MessagePassing
from torch_geometric.utils  import dropout_edge, add_self_loops, softmax


# ==============================================================================
# PROJECTION LAYER  (14 → hidden_dim)
# ==============================================================================
class ProjectionLayer(nn.Module):
    """Linear(14→D) → LayerNorm(D) → ELU  — tránh Feature Domination."""

    def __init__(self, in_dim: int = 14, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ==============================================================================
# [M1]  DEEP GAT BLOCK với Edge-Weight
# ==============================================================================
class DeepGATBlock(nn.Module):
    """
    1 Block = DropEdge → GATv2Conv(edge_attr) → Residual → LayerNorm → ELU

    [M1] GATv2Conv nhận `edge_attr` (scalar edge_weight được mở rộng thành
         vector dim=1 rồi chiếu bởi GATv2Conv qua tham số `edge_dim=1`).
         Điều này cho phép mô hình điều chỉnh trọng số attention theo
         độ tin cậy của từng cạnh trong đồ thị HumanNet.
    """

    def __init__(self,
                 hidden_dim:   int   = 64,
                 heads:        int   = 4,
                 dropout:      float = 0.3,
                 drop_edge_p:  float = 0.2):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim phải chia hết cho heads"
        self.drop_edge_p = drop_edge_p

        self.conv = GATv2Conv(
            in_channels  = hidden_dim,
            out_channels = hidden_dim // heads,
            heads        = heads,
            concat       = True,      # output = hidden_dim // heads * heads = hidden_dim
            dropout      = dropout,
            edge_dim     = 1,         # [M1] scalar edge_weight được mở rộng thành (E,1)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.ELU()

    def forward(self,
                x:           torch.Tensor,
                edge_index:  torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x           : [N, hidden_dim]
            edge_index  : [2, E]
            edge_weight : [E]   — scalar, đã chuẩn hoá về [0,1]
        """
        # DropEdge khử nhiễu topology khi train
        if self.training:
            edge_index, mask = dropout_edge(
                edge_index, p=self.drop_edge_p, training=True
            )
            edge_weight = edge_weight[mask]

        # GATv2Conv yêu cầu edge_attr shape [E, edge_dim]
        edge_attr = edge_weight.unsqueeze(-1)   # [E, 1]

        out = self.conv(x, edge_index, edge_attr=edge_attr)   # [N, hidden_dim]
        out = self.norm(out + x)                              # Residual
        out = self.act(out)
        return out


# ==============================================================================
# [M3]  SPARSE CROSS-ATTENTION via MessagePassing
# ==============================================================================
class SparseCrossAttention(MessagePassing):
    """
    Graph-aware Cross-Attention: Gen i ở Normal chỉ attend với Gen j ở Tumor
    nếu (i, j) tồn tại trong edge_index (+ self-loop).

    Độ phức tạp: O(E * D)  thay vì O(N² * D)  →  không OOM với N=9000.

    Cơ chế:
        score(i,j) = (Q_i · K_j) / sqrt(D/heads)
        α(i,j)     = softmax trên toàn bộ hàng xóm j của i  (edge-level)
        out_i      = Σ_j  α(i,j) * V_j

    Sau đó ghép với Residual Hint: Z_diff = (Z_n − Z_t)²
    và chiếu qua Linear(2D → D).
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        # aggr='add' vì chúng ta tự normalize bằng softmax rồi
        super().__init__(aggr='add',node_dim=0)
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        # Projection cho Q (từ Normal), K và V (từ Tumor)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Fusion: concat(Z_attn [D], Z_diff [D]) → D
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self,
                z_n:         torch.Tensor,
                z_t:         torch.Tensor,
                edge_index:  torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_n        : [N, D]  — embedding Normal   (Query)
            z_t        : [N, D]  — embedding Tumor    (Key / Value)
            edge_index : [2, E]  — đồ thị chia sẻ (sẽ thêm self-loop)
        Returns:
            z_fusion   : [N, D]
        """
        N = z_n.size(0)

        # Thêm self-loop để gen i luôn attend với chính nó
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=N)

        # Project → tách heads: [N, heads, head_dim]
        Q = self.W_q(z_n).view(N, self.num_heads, self.head_dim)
        K = self.W_k(z_t).view(N, self.num_heads, self.head_dim)
        V = self.W_v(z_t).view(N, self.num_heads, self.head_dim)

        # MessagePassing: propagate gửi (K_j, V_j) tới node i
        # Lưu Q, K, V vào self để dùng trong message()
        self._Q = Q
        self._K = K
        self._V = V

        # z_attn: [N, heads*head_dim] = [N, D]
        z_attn = self.propagate(edge_index_sl, size=(N, N),
                                Q=Q, K=K, V=V)
        z_attn = self.out_proj(z_attn)   # [N, D]

        # Residual Hint: bình phương khoảng cách tiềm ẩn
        z_diff   = (z_n - z_t) ** 2      # [N, D]

        z_fusion = torch.cat([z_attn, z_diff], dim=-1)   # [N, 2D]
        z_fusion = self.fusion_proj(z_fusion)             # [N, D]
        z_fusion = self.norm(z_fusion)
        return z_fusion

    def message(self,
                Q_i: torch.Tensor,
                K_j: torch.Tensor,
                V_j: torch.Tensor,
                index: torch.Tensor,
                size_i: int) -> torch.Tensor:
        """
        Tính attention score và trả về α_ij * V_j cho từng cạnh.

        Shapes:
            Q_i, K_j, V_j : [E, heads, head_dim]
            index          : [E]   — node đích (= i)
        """
        # Dot-product score: [E, heads]
        score = (Q_i * K_j).sum(dim=-1) * self.scale     # [E, heads]

        # Softmax theo hàng xóm của mỗi node i (sparse softmax)
        alpha = softmax(score, index, num_nodes=size_i)    # [E, heads]

        # Áp trọng số attention lên Value
        # alpha: [E, heads, 1]  *  V_j: [E, heads, head_dim] → [E, heads, head_dim]
        out = alpha.unsqueeze(-1) * V_j                    # [E, heads, head_dim]

        # Flatten heads: [E, D]
        return out.view(out.size(0), -1)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        # aggr_out: [N, D]  (đã sum qua tất cả hàng xóm)
        return aggr_out


# ==============================================================================
# [M2]  JUMPING KNOWLEDGE AGGREGATOR
# ==============================================================================
class JKNetAggregator(nn.Module):
    """
    Nối output của 3 lớp GAT: concat([z1, z2, z3]) → Linear(3D → D).
    Giữ lại đặc trưng cục bộ (lớp 1) lẫn toàn cục (lớp 3), chống over-smoothing.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

    def forward(self, layer_outputs: list) -> torch.Tensor:
        """
        Args:
            layer_outputs: list of Tensor [N, hidden_dim], len = num_layers
        Returns:
            z_jk: [N, hidden_dim]
        """
        return self.proj(torch.cat(layer_outputs, dim=-1))


# ==============================================================================
# MÔ HÌNH CHÍNH: SiamesePUGNN v2
# ==============================================================================
class SiamesePUGNN(nn.Module):
    """
    Pipeline đầy đủ:

        x_n, x_t
          │
          ▼  Projection(14 → D)
        h_n, h_t
          │
          ▼  Siamese DeepGATBlock × 3  (shared weights, edge_attr)  [M1]
        [z_n1,z_n2,z_n3], [z_t1,z_t2,z_t3]
          │
          ▼  JK-Net concat + proj  [M2]
        Z_n [N,D], Z_t [N,D]
          │
          ▼  SparseCrossAttention (edge-conditioned)  [M3]
        Z_fusion [N,D]
          │
          ▼  Classifier (Linear→BN→ELU→Dropout→Linear)
        logits [N]
    """

    def __init__(self,
                 in_dim:        int   = 14,
                 hidden_dim:    int   = 64,
                 heads:         int   = 4,
                 num_gat_layers:int   = 3,
                 dropout:       float = 0.3,
                 drop_edge_p:   float = 0.2):
        super().__init__()
        self.num_gat_layers = num_gat_layers

        # Projection (shared)
        self.projection = ProjectionLayer(in_dim=in_dim, out_dim=hidden_dim)

        # Siamese GAT blocks (shared weights giữa nhánh Normal & Tumor)
        self.gat_blocks = nn.ModuleList([
            DeepGATBlock(
                hidden_dim  = hidden_dim,
                heads       = heads,
                dropout     = dropout,
                drop_edge_p = drop_edge_p,
            )
            for _ in range(num_gat_layers)
        ])

        # JK-Net aggregator (1 cái dùng chung cho cả 2 nhánh — shared weights)
        self.jk_agg = JKNetAggregator(
            hidden_dim = hidden_dim,
            num_layers = num_gat_layers,
        )

        # Sparse Cross-Attention Fusion
        self.cross_attn = SparseCrossAttention(
            embed_dim = hidden_dim,
            num_heads = heads,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            # KHÔNG Sigmoid — BCEWithLogitsLoss trong nnPU tự xử lý
        )

    def encode(self,
               x:           torch.Tensor,
               edge_index:  torch.Tensor,
               edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Mã hoá 1 nhánh qua Projection → 3 DeepGATBlock → JK-Net.

        Returns: [N, hidden_dim]
        """
        h = self.projection(x)
        layer_outs = []
        for block in self.gat_blocks:
            h = block(h, edge_index, edge_weight)
            layer_outs.append(h)
        z = self.jk_agg(layer_outs)   # [M2] JK-Net concat
        return z

    def forward(self,
                x_n:  torch.Tensor,
                x_t:  torch.Tensor,
                e_n:  torch.Tensor,
                ew_n: torch.Tensor,
                e_t:  torch.Tensor  = None,
                ew_t: torch.Tensor  = None) -> torch.Tensor:
        """
        Args:
            x_n  : [N, in_dim]
            x_t  : [N, in_dim]
            e_n  : [2, E_n]   — edge index Normal
            ew_n : [E_n]      — edge weight Normal
            e_t  : [2, E_t]   — edge index Tumor  (None → dùng e_n)
            ew_t : [E_t]      — edge weight Tumor  (None → dùng ew_n)
        Returns:
            logits : [N]
        """
        if e_t  is None: e_t  = e_n
        if ew_t is None: ew_t = ew_n

        # ── Siamese Encoding ──────────────────────────────────────────────
        z_n = self.encode(x_n, e_n, ew_n)   # [N, D]
        z_t = self.encode(x_t, e_t, ew_t)   # [N, D]

        # ── Sparse Cross-Attention ────────────────────────────────────────
        # Dùng e_n làm đồ thị tham chiếu (có thể đổi thành union của e_n & e_t)
        z_fusion = self.cross_attn(z_n, z_t, e_n)   # [N, D]

        # ── Classification ────────────────────────────────────────────────
        logits = self.classifier(z_fusion).squeeze(-1)   # [N]
        return logits


# ==============================================================================
# ROBUST nnPU LOSS  (kế thừa nguyên vẹn, chống NaN + gradient flip)
# ==============================================================================
def robust_nnpu_loss(logits:  torch.Tensor,
                     labels:  torch.Tensor,
                     prior:   float = 0.05) -> torch.Tensor:
    """
    Non-negative PU Loss bền vững với gradient.

    Công thức (du Plessis et al., 2015):
        R_PU(f) = π * R_P+(f)  +  max(0, R_U-(f) - π * R_P-(f))

    Kỹ thuật chống NaN / gradient bùng nổ:
        • Tất cả BCE tính qua F.binary_cross_entropy_with_logits (numerically stable).
        • Nếu risk_neg < -1e-4 → lật dấu (flip sign) để gradient vẫn truyền
          được nhưng theo hướng đúng, tránh collapse.
        • Gradient clipping nằm ở run_model.py (max_norm=1.0).

    Args:
        logits : Tensor [N]  — raw logits chưa sigmoid
        labels : Tensor [N]  — 1.0=known Positive, 0.0=Unlabeled
        prior  : float       — π = P(Y=1), ví dụ từ AlphaMax

    Returns:
        objective : scalar Tensor
    """
    pos_mask = (labels == 1.0).float()
    unl_mask = (labels == 0.0).float()

    num_pos = max(pos_mask.sum().item(), 1.0)
    num_unl = max(unl_mask.sum().item(), 1.0)

    loss_p = F.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits),  reduction='none')
    loss_n = F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction='none')

    term_pos     = (loss_p * pos_mask).sum() / num_pos   # E_P[l(f,+1)]
    term_unl_neg = (loss_n * unl_mask).sum() / num_unl   # E_U[l(f,-1)]
    term_pos_neg = (loss_n * pos_mask).sum() / num_pos   # E_P[l(f,-1)]

    risk_neg = term_unl_neg - prior * term_pos_neg

    if risk_neg.item() < -1e-4:
        # Gradient flip: mô hình học để đẩy risk_neg lên 0, không xuống âm vô hạn
        objective = prior * term_pos + (-risk_neg)
    else:
        objective = prior * term_pos + risk_neg

    return objective