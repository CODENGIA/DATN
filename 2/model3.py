import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# ==============================================================================
# MODULE 1: ENCODER — Giữ kiến trúc GAT + Cross-Attention từ file gốc,
#           nhưng nhận đầu vào là x đã bị mask (một số node bị thay bằng 0)
# ==============================================================================
class DualGATEncoder(nn.Module):
    """
    Encoder GATConv cho 1 đồ thị (Normal hoặc Tumor).
    Giữ nguyên cấu trúc 2 lớp GAT từ file gốc.
    hidden_dim=32, heads=4 → 128 chiều → nén về embed_dim=64.
    """
    def __init__(self, input_dim=14, hidden_dim=32, embed_dim=64, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, embed_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h   # (N, embed_dim)

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention giữa Normal và Tumor — giữ nguyên từ file gốc.
    Normal query Tumor và Tumor query Normal.
    """
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.2):
        super().__init__()
        self.attn_n2t = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1    = nn.LayerNorm(embed_dim)
        self.attn_t2n = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2    = nn.LayerNorm(embed_dim)

    def forward(self, z_n, z_t):
        zn_q  = z_n.unsqueeze(0)   # (1, N, D)
        zt_kv = z_t.unsqueeze(0)
        
        out_n2t, _ = self.attn_n2t(query=zn_q, key=zt_kv, value=zt_kv, need_weights=False)
        fused_n    = self.norm1(zn_q + out_n2t).squeeze(0)   # (N, D)
        
        out_t2n, _ = self.attn_t2n(query=zt_kv, key=zn_q, value=zn_q, need_weights=False)
        fused_t    = self.norm2(zt_kv + out_t2n).squeeze(0)  # (N, D)
        
        return fused_n, fused_t

# ==============================================================================
# MODULE 2: DECODER — ScaledCosineDecoder theo GraphMAE
#           Tái tạo đặc trưng gốc từ vector embedding.
#           Dùng Cosine Loss thay MSE để tránh vấn đề scale.
# ==============================================================================
class FeatureDecoder(nn.Module):
    """
    Decoder 2 lớp Linear đơn giản.
    embed_dim → hidden → input_dim (tái tạo 14 chiều ban đầu).
    """
    def __init__(self, embed_dim=64, hidden_dim=32, output_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.net(z)   # (N, input_dim)

# ==============================================================================
# MODULE 3: MÔ HÌNH MGAE CHÍNH
#           Ghép Encoder → CrossAttention → Decoder thành 1 khối.
# ==============================================================================
class MGAEDualGraphModel(nn.Module):
    """
    Masked Graph AutoEncoder cho 2 đồ thị song song (Normal + Tumor).
    Luồng training:
      1. Mask ngẫu nhiên ~mask_ratio node trên cả 2 đồ thị (đặt feature = 0).
      2. Encoder GATConv mã hóa đồ thị đã mask → z_n, z_t.
      3. CrossAttention để 2 đồ thị học ngữ cảnh của nhau → fused_n, fused_t.
      4. Decoder tái tạo lại feature gốc của các node bị mask.
      5. Loss = CosineLoss chỉ trên các node bị mask.
    Luồng inference:
      - Không mask (toàn bộ node).
      - Anomaly Score = reconstruction error trung bình Normal + Tumor.
      - Gen bệnh có đặc trưng dị biệt → error cao → điểm bất thường cao.
    """
    def __init__(self, input_dim=14, embed_dim=64, num_heads=4,
                 decoder_hidden=32, dropout=0.2, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio  = mask_ratio
        self.input_dim   = input_dim
        
        # Encoder độc lập cho Normal và Tumor (như file gốc)
        self.encoder_n = DualGATEncoder(input_dim, hidden_dim=32, embed_dim=embed_dim, dropout=dropout)
        self.encoder_t = DualGATEncoder(input_dim, hidden_dim=32, embed_dim=embed_dim, dropout=dropout)
        
        # Cross-Attention (giữ nguyên thiết kế từ file gốc)
        self.cross_attn = CrossAttentionFusion(embed_dim, num_heads, dropout)
        
        # Decoder tái tạo feature (MỚI — thay InfoNCE)
        self.decoder_n = FeatureDecoder(embed_dim, decoder_hidden, input_dim)
        self.decoder_t = FeatureDecoder(embed_dim, decoder_hidden, input_dim)

    # ------------------------------------------------------------------
    # Hàm tạo mask ngẫu nhiên: trả về boolean tensor (N,)
    # True = node bị mask (cần tái tạo), False = node còn nguyên
    # ------------------------------------------------------------------
    def _make_mask(self, num_nodes, device):
        num_masked = max(1, int(num_nodes * self.mask_ratio))
        perm       = torch.randperm(num_nodes, device=device)
        mask       = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask[perm[:num_masked]] = True
        return mask   # (N,)

    # ------------------------------------------------------------------
    # Forward dùng khi TRAINING: có mask
    # Trả về (recon_n, recon_t, mask_n, mask_t, x_n_orig, x_t_orig)
    # ------------------------------------------------------------------
    def forward_train(self, x_n, edge_n, x_t, edge_t):
        N = x_n.size(0)
        
        # 1. Tạo mask độc lập cho Normal và Tumor
        mask_n = self._make_mask(N, x_n.device)   # (N,) bool
        mask_t = self._make_mask(N, x_t.device)
        
        # 2. Áp mask: thay feature node bị mask bằng vector 0
        x_n_masked = x_n.clone()
        x_t_masked = x_t.clone()
        x_n_masked[mask_n] = 0.0
        x_t_masked[mask_t] = 0.0
        
        # 3. Encode đồ thị đã mask
        z_n = self.encoder_n(x_n_masked, edge_n)   # (N, embed_dim)
        z_t = self.encoder_t(x_t_masked, edge_t)
        
        # 4. Cross-Attention
        fused_n, fused_t = self.cross_attn(z_n, z_t)
        
        # 5. Decode: tái tạo feature gốc
        recon_n = self.decoder_n(fused_n)   # (N, input_dim)
        recon_t = self.decoder_t(fused_t)
        
        return recon_n, recon_t, mask_n, mask_t

    # ------------------------------------------------------------------
    # Forward dùng khi INFERENCE: không mask, lấy reconstruction error
    # ------------------------------------------------------------------
    def forward_inference(self, x_n, edge_n, x_t, edge_t):
        z_n = self.encoder_n(x_n, edge_n)
        z_t = self.encoder_t(x_t, edge_t)
        
        fused_n, fused_t = self.cross_attn(z_n, z_t)
        
        recon_n = self.decoder_n(fused_n)
        recon_t = self.decoder_t(fused_t)
        
        return recon_n, recon_t

# ==============================================================================
# MODULE 4: HÀM LOSS — Scaled Cosine Error (theo GraphMAE)
#           Chỉ tính loss trên các node bị mask.
#           Dùng Cosine thay MSE để tránh mất cân bằng scale giữa 14 features.
# ==============================================================================
def mgae_loss(x_original, x_reconstructed, mask):
    """
    Scaled Cosine Reconstruction Loss — chỉ tính trên node bị mask.
    Công thức:
        L = mean over masked nodes of [ 1 - cosine_similarity(x_i, x̂_i) ]
    x_original      : (N, F) — feature gốc
    x_reconstructed : (N, F) — feature tái tạo từ decoder
    mask            : (N,)   — True = node bị mask
    Trả về: scalar loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x_original.device, requires_grad=True)
        
    x_orig_masked  = x_original[mask]        # (M, F)
    x_recon_masked = x_reconstructed[mask]   # (M, F)
    
    # Cosine similarity per node → (M,)
    cos_sim = F.cosine_similarity(x_orig_masked, x_recon_masked, dim=1)
    
    # Loss = 1 - similarity (cao nghĩa là tái tạo tệ)
    loss = (1.0 - cos_sim).mean()
    return loss

def anomaly_score_per_node(x_original, x_reconstructed):
    """
    Tính reconstruction error cho TỪNG node (dùng lúc inference).
    Dùng MSE per node để có granularity cao hơn Cosine khi ranking.
    Trả về: (N,) numpy array — error của từng node
    """
    mse = torch.mean((x_original - x_reconstructed) ** 2, dim=1)  # (N,)
    return mse.cpu().numpy()