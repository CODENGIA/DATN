import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj, negative_sampling

# Hàm che khuất đặc trưng giả lập lỗi giải trình tự (Data Augmentation)
def mask_node_features(x, p=0.2, training=True):
    if not training or p == 0.0:
        return x
    mask = torch.rand(x.shape, device=x.device) > p
    return x * mask / (1.0 - p)

class DualHybridFeatureExtractor(nn.Module):
    """
    Khối trích xuất đặc trưng lai GCN + MLP chạy song song.
    """
    def __init__(self, input_dim=14, hidden_dim=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        # Nhánh Mạng Normal (Khỏe mạnh)
        self.norm_fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm_gcn1 = GCNConv(input_dim, hidden_dim, add_self_loops=False)
        self.norm_fc2 = nn.Linear(2 * hidden_dim, out_dim)
        self.norm_gcn2 = GCNConv(2 * hidden_dim, out_dim, add_self_loops=False)
        
        # Nhánh Mạng Tumor (Ung thư)
        self.tumor_fc1 = nn.Linear(input_dim, hidden_dim)
        self.tumor_gcn1 = GCNConv(input_dim, hidden_dim, add_self_loops=False)
        self.tumor_fc2 = nn.Linear(2 * hidden_dim, out_dim)
        self.tumor_gcn2 = GCNConv(2 * hidden_dim, out_dim, add_self_loops=False)

    def forward(self, x_n, e_n, x_t, e_t):
        # Tăng cường dữ liệu (Augmentation)
        x_n_aug = mask_node_features(x_n, p=0.1, training=self.training)
        x_t_aug = mask_node_features(x_t, p=0.25, training=self.training)
        e_n_aug, _ = dropout_adj(e_n, p=0.15, training=self.training)
        e_t_aug, _ = dropout_adj(e_t, p=0.15, training=self.training)

        # --- XỬ LÝ NHÁNH NORMAL ---
        zn = F.dropout(x_n_aug, p=self.dropout, training=self.training)
        zn_mlp1 = F.relu(self.norm_fc1(zn))
        zn_gcn1 = F.relu(self.norm_gcn1(zn, e_n_aug))
        zn_cat = torch.cat((zn_mlp1, zn_gcn1), dim=1) # Nối đặc trưng
        
        zn_cat = F.dropout(zn_cat, p=self.dropout, training=self.training)
        zn_mlp2 = F.relu(self.norm_fc2(zn_cat))
        zn_gcn2 = F.relu(self.norm_gcn2(zn_cat, e_n_aug))
        z_n = zn_mlp2 + zn_gcn2 # Cộng dồn đặc trưng
        
        # --- XỬ LÝ NHÁNH TUMOR ---
        zt = F.dropout(x_t_aug, p=self.dropout, training=self.training)
        zt_mlp1 = F.relu(self.tumor_fc1(zt))
        zt_gcn1 = F.relu(self.tumor_gcn1(zt, e_t_aug))
        zt_cat = torch.cat((zt_mlp1, zt_gcn1), dim=1) 
        
        zt_cat = F.dropout(zt_cat, p=self.dropout, training=self.training)
        zt_mlp2 = F.relu(self.tumor_fc2(zt_cat))
        zt_gcn2 = F.relu(self.tumor_gcn2(zt_cat, e_t_aug))
        z_t = zt_mlp2 + zt_gcn2 

        return z_n, z_t

class DualGraphCrossAttentionModel(nn.Module):
    def __init__(self, input_dim=14, embed_dim=128, num_heads=4, dropout=0.2):
        super().__init__()
        self.hybrid_extractor = DualHybridFeatureExtractor(input_dim, hidden_dim=64, out_dim=embed_dim, dropout=dropout)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        
        self.cross_attn_rev = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x_n, e_n, x_t, e_t):
        z_n, z_t = self.hybrid_extractor(x_n, e_n, x_t, e_t)
        
        zn_q, zt_kv = z_n.unsqueeze(0), z_t.unsqueeze(0)
        
        attn_n2t, _ = self.cross_attn(query=zn_q, key=zt_kv, value=zt_kv, need_weights=False)
        out_n = self.layer_norm1(zn_q + attn_n2t).squeeze(0)
        
        attn_t2n, _ = self.cross_attn_rev(query=zt_kv, key=zn_q, value=zn_q, need_weights=False)
        out_t = self.layer_norm2(zt_kv + attn_t2n).squeeze(0)
        
        return out_n, out_t

# --- HÀM TÍNH LOSS KHÔNG GIÁM SÁT ---

def info_nce_loss(out_n, out_t, temperature=0.1, num_negatives=5, return_individual_loss=False):
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

def edge_reconstruction_loss(z, pos_edge_index, pos_weight_val):
    pos_logits = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    pos_labels = torch.ones_like(pos_logits)
    
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index, 
        num_nodes=z.size(0),
        num_neg_samples=pos_edge_index.size(1)
    )
    neg_logits = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    neg_labels = torch.zeros_like(neg_logits)
    
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([pos_labels, neg_labels])
    
    weight_tensor = torch.cat([
        torch.full_like(pos_labels, pos_weight_val),
        torch.ones_like(neg_labels)
    ])
    
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weight_tensor)