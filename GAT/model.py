import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_adj, negative_sampling

def mask_node_features(x, p=0.2, training=True):
    if not training or p == 0.0:
        return x
    mask = torch.rand(x.shape, device=x.device) > p
    return x * mask / (1.0 - p)

class PureDualGraphAutoencoder(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        # Nhánh Normal dùng GAT để bắt cấu trúc (Attention giúp tự động bỏ qua nhiễu rác)
        self.norm_conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.norm_conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False)
        
        # Nhánh Tumor dùng GAT
        self.tumor_conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.tumor_conv2 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False)

    def forward(self, x_n, e_n, x_t, e_t):
        # Tăng cường dữ liệu (Vẫn giữ để tránh học vẹt)
        x_n_aug = mask_node_features(x_n, p=0.05, training=self.training)
        x_t_aug = mask_node_features(x_t, p=0.20, training=self.training)
        e_n_aug, _ = dropout_adj(e_n, p=0.05, training=self.training)
        e_t_aug, _ = dropout_adj(e_t, p=0.20, training=self.training)

        # Trích xuất vector mạng Normal
        z_n = F.dropout(x_n_aug, p=self.dropout, training=self.training)
        z_n = F.elu(self.norm_conv1(z_n, e_n_aug))
        z_n = F.dropout(z_n, p=self.dropout, training=self.training)
        out_n = self.norm_conv2(z_n, e_n_aug)
        
        # Trích xuất vector mạng Tumor
        z_t = F.dropout(x_t_aug, p=self.dropout, training=self.training)
        z_t = F.elu(self.tumor_conv1(z_t, e_t_aug))
        z_t = F.dropout(z_t, p=self.dropout, training=self.training)
        out_t = self.tumor_conv2(z_t, e_t_aug)
        
        # Trả thẳng đầu ra, ĐÃ BỎ HOÀN TOÀN CROSS-ATTENTION
        return out_n, out_t

# ĐÃ BỎ HÀM INFONCE_LOSS
# Chỉ giữ lại hàm Loss tái tạo đồ thị làm động lực huấn luyện duy nhất
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