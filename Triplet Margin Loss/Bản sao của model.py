import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DualGNNFeatureExtractor(nn.Module):
    """Multiple View: Trích xuất đặc trưng độc lập cho trạng thái Normal và Tumor"""
    def __init__(self, input_dim=14, hidden_dim=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.norm_conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.norm_conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False)
        
        self.tumor_conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.tumor_conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False)

    def forward(self, x_n, e_n, x_t, e_t):
        z_n = F.dropout(x_n, p=self.dropout, training=self.training)
        z_n = F.elu(self.norm_conv1(z_n, e_n))
        z_n = F.dropout(z_n, p=self.dropout, training=self.training)
        z_n = self.norm_conv2(z_n, e_n)
        
        z_t = F.dropout(x_t, p=self.dropout, training=self.training)
        z_t = F.elu(self.tumor_conv1(z_t, e_t))
        z_t = F.dropout(z_t, p=self.dropout, training=self.training)
        z_t = self.tumor_conv2(z_t, e_t)
        
        return z_n, z_t

class DualGraphCrossAttentionModel(nn.Module):
    def __init__(self, input_dim=14, embed_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.gnn_extractor = DualGNNFeatureExtractor(input_dim, hidden_dim=32, out_dim=embed_dim, dropout=dropout)
        
        # Cross Attention: Học hỏi thông tin chéo giữa 2 View
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        self.cross_attn_rev = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x_n, e_n, x_t, e_t):
        z_n, z_t = self.gnn_extractor(x_n, e_n, x_t, e_t)
        
        zn_q, zt_kv = z_n.unsqueeze(0), z_t.unsqueeze(0)
        attn_n2t, _ = self.cross_attn(query=zn_q, key=zt_kv, value=zt_kv,need_weights=False)
        out_n = self.layer_norm1(zn_q + attn_n2t).squeeze(0)
        
        attn_t2n, _ = self.cross_attn_rev(query=zt_kv, key=zn_q, value=zn_q,need_weights=False)
        out_t = self.layer_norm2(zt_kv + attn_t2n).squeeze(0)
        
        # Trả về 2 không gian vector độc lập (Không gộp lại thành 1)
        return out_n, out_t

def self_supervised_contrastive_loss(out_n, out_t, margin=0.5):
    """
    Ép mô hình phải nhận diện được Tumor của chính nó, thay vì Tumor của gen khác.
    """
    # 1. Chuẩn hóa vector về cùng không gian (Rất quan trọng để tránh bùng nổ Loss)
    out_n = F.normalize(out_n, p=2, dim=1)
    out_t = F.normalize(out_t, p=2, dim=1)
    
    # 2. Positive Pair (Mẫu dương): Độ tương đồng Cosine giữa Normal và Tumor của CÙNG 1 GEN
    pos_sim = torch.sum(out_n * out_t, dim=1) 
    
    # 3. Negative Pair (Mẫu âm): Độ tương đồng giữa Normal gen này và Tumor gen KHÁC
    idx_shuff = torch.randperm(out_n.size(0))
    neg_sim = torch.sum(out_n * out_t[idx_shuff], dim=1)
    
    # 4. Hàm Margin Loss: Ép pos_sim phải lớn hơn neg_sim ít nhất một khoảng là margin
    # Nếu mô hình lười biếng cho tất cả vector giống nhau (pos_sim = neg_sim), nó sẽ bị phạt nặng.
    target = torch.ones_like(pos_sim)
    loss = F.margin_ranking_loss(pos_sim, neg_sim, target, margin=margin)
    
    return loss