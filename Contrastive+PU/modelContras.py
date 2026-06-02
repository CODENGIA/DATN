import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class ModernPUGNN(nn.Module):
    """
    Kiến trúc Late Fusion GATv2 kết hợp Residual Skip-Connection.
    Hỗ trợ Contrastive Learning (Siamese) ở Stage 1 và PU Learning ở Stage 2.
    """
    def __init__(self, in_dim=14, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        # BỘ MÃ HÓA (ENCODER) - Học chung trọng số cho cả Normal và Tumor
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        
        # BỘ PHÂN LOẠI (CLASSIFIER) - Dùng cho Stage 2
        # Đầu vào: [z_n (hidden_dim) + z_t (hidden_dim) + raw_diff (in_dim)]
        classifier_in_dim = hidden_dim * 2 + in_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 64),
            nn.BatchNorm1d(64), # Giữ cho phân phối không bị xẹp về 0.04
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def get_embedding(self, x, edge_index):
        """
        Hàm trích xuất đặc trưng cho giai đoạn Contrastive Pre-training (Giấu nhãn)
        """
        z = F.dropout(x, p=self.dropout, training=self.training)
        z = F.elu(self.conv1(z, edge_index))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = F.elu(self.conv2(z, edge_index))
        return z

    def forward(self, x_n, x_t, edge_index_n, edge_index_t):
        # 1. Truyền độc lập qua GNN để lấy tri thức mạng lưới
        z_n = self.get_embedding(x_n, edge_index_n)
        z_t = self.get_embedding(x_t, edge_index_t)
        
        # 2. ĐƯỜNG TẮT (Skip-Connection): Lấy độ lệch thuần túy của biểu hiện gen
        raw_diff = torch.abs(x_n - x_t)
        
        # 3. Gộp tri thức mạng lưới (Normal & Tumor) và tri thức sinh học độc lập
        z_final = torch.cat([z_n, z_t, raw_diff], dim=-1)
        
        # 4. Phân loại
        logits = self.classifier(z_final).squeeze(-1)
        return logits

def nnpu_loss(logits, labels, prior=0.05):
    """
    Non-negative PU Loss kinh điển.
    """
    pos_mask = (labels == 1.0).float()
    unl_mask = (labels == 0.0).float()
    
    num_pos = max(pos_mask.sum().item(), 1.0)
    num_unl = max(unl_mask.sum().item(), 1.0)
    
    loss_pos = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction='none')
    term_pos = (loss_pos * pos_mask).sum() / num_pos
    
    loss_unl_neg = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
    term_unl_neg = (loss_unl_neg * unl_mask).sum() / num_unl
    
    loss_pos_neg = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction='none')
    term_pos_neg = (loss_pos_neg * pos_mask).sum() / num_pos
    
    risk_neg = term_unl_neg - prior * term_pos_neg
    
    # Kích hoạt Non-negative
    if risk_neg < 0:
        objective = prior * term_pos - risk_neg
    else:
        objective = prior * term_pos + risk_neg
        
    return objective