import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_adj, negative_sampling

# [NÂNG CẤP 1]: Hàm che khuất đặc trưng giả lập lỗi giải trình tự (Omics/Mutation Masking)
def mask_node_features(x, p=0.2, training=True):
    if not training or p == 0.0:
        return x
    # Tạo mask ngẫu nhiên, che đi p% đặc trưng
    mask = torch.rand(x.shape, device=x.device) > p
    return x * mask / (1.0 - p) # Scale lại để bù đắp giá trị bị mất

class DualGNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.norm_conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.norm_conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False)
        
        self.tumor_conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.tumor_conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False)

    def forward(self, x_n, e_n, x_t, e_t):
        # [NÂNG CẤP 1]: Data Augmentation - Masking Node & Edge Dropout
        x_n_aug = mask_node_features(x_n, p=0.1, training=self.training)
        x_t_aug = mask_node_features(x_t, p=0.15, training=self.training) # Tumor mask mạnh hơn vì nhiễu nhiều hơn
        
        e_n_aug, _ = dropout_adj(e_n, p=0.15, training=self.training)
        e_t_aug, _ = dropout_adj(e_t, p=0.15, training=self.training)

        # Trích xuất đặc trưng Mạng Normal
        z_n = F.dropout(x_n_aug, p=self.dropout, training=self.training)
        z_n = F.elu(self.norm_conv1(z_n, e_n_aug))
        z_n = F.dropout(z_n, p=self.dropout, training=self.training)
        z_n = self.norm_conv2(z_n, e_n_aug)
        
        # Trích xuất đặc trưng Mạng Tumor
        z_t = F.dropout(x_t_aug, p=self.dropout, training=self.training)
        z_t = F.elu(self.tumor_conv1(z_t, e_t_aug))
        z_t = F.dropout(z_t, p=self.dropout, training=self.training)
        z_t = self.tumor_conv2(z_t, e_t_aug)
        
        return z_n, z_t

class DualGraphCrossAttentionModel(nn.Module):
    def __init__(self, input_dim=14, embed_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.gnn_extractor = DualGNNFeatureExtractor(input_dim, hidden_dim=32, out_dim=embed_dim, dropout=dropout)
        
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        self.cross_attn_rev = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # [NÂNG CẤP 2]: BỎ FeatureReconHead. Việc tái tạo cấu trúc đồ thị sẽ dùng Tích vô hướng trực tiếp ở hàm Loss

    def forward(self, x_n, e_n, x_t, e_t):
        z_n, z_t = self.gnn_extractor(x_n, e_n, x_t, e_t)
        
        zn_q, zt_kv = z_n.unsqueeze(0), z_t.unsqueeze(0)
        
        # Normal học hỏi từ Tumor
        attn_n2t, _ = self.cross_attn(query=zn_q, key=zt_kv, value=zt_kv, need_weights=False)
        out_n = self.layer_norm1(zn_q + attn_n2t).squeeze(0)
        
        # Tumor học hỏi từ Normal
        attn_t2n, _ = self.cross_attn_rev(query=zt_kv, key=zn_q, value=zn_q, need_weights=False)
        out_t = self.layer_norm2(zt_kv + attn_t2n).squeeze(0)
        
        return z_n, z_t

# --- CÁC HÀM TÍNH LOSS ---

def info_nce_loss(out_n, out_t, temperature=0.1, num_negatives=5, return_individual_loss=False):
    """Loss InfoNCE để đẩy vector Normal và Tumor của cùng 1 gen lại gần nhau, đẩy gen khác ra xa"""
    out_n = F.normalize(out_n, p=2, dim=1)
    out_t = F.normalize(out_t, p=2, dim=1)
    batch_size = out_n.size(0)
    
    pos_sim = torch.sum(out_n * out_t, dim=1, keepdim=True) 
    neg_sims = []
    for _ in range(num_negatives):
        idx_shuff = torch.randperm(batch_size, device=out_n.device)
        neg_sim = torch.sum(out_n * out_t[idx_shuff], dim=1, keepdim=True)
        neg_sims.append(neg_sim)
        
    neg_sims = torch.cat(neg_sims, dim=1)
    logits = torch.cat([pos_sim, neg_sims], dim=1) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=out_n.device)
    
    if return_individual_loss:
        return F.cross_entropy(logits, labels, reduction='none')
    return F.cross_entropy(logits, labels)

# [NÂNG CẤP 2 & 3]: Hàm tái tạo cấu trúc Cạnh (Edge Reconstruction Loss) tích hợp pos_weight
def edge_reconstruction_loss(z, pos_edge_index, pos_weight_val):
    """
    Tính loss dự đoán cạnh để khôi phục lại cấu trúc mạng lưới PPI.
    """
    # 1. Tính Logits cho các cạnh thực tế (Positive edges)
    pos_logits = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    pos_labels = torch.ones_like(pos_logits)
    
    # 2. Lấy mẫu các cạnh không tồn tại (Negative edges) để làm mồi nhử
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index, 
        num_nodes=z.size(0),
        num_neg_samples=pos_edge_index.size(1) # Lấy mẫu theo tỉ lệ 1:1
    )
    neg_logits = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    neg_labels = torch.zeros_like(neg_logits)
    
    # Gộp chung
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([pos_labels, neg_labels])
    
    # [NÂNG CẤP 3]: Áp dụng pos_weight để phạt nặng nếu mô hình bỏ sót cạnh có thật
    weight_tensor = torch.cat([
        torch.full_like(pos_labels, pos_weight_val), # Trọng số lớn cho cạnh thật
        torch.ones_like(neg_labels)                  # Trọng số 1 cho cạnh giả
    ])
    
    loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weight_tensor)
    return loss