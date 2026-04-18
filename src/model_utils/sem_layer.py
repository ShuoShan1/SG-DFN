from torch_scatter import scatter_sum
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add, scatter_mean



# ==========================语义编码器==========================
class Deep_Residual_Layer(nn.Module):
    def __init__(self, in_features=1024,out_features=100,depth=2,dropout=0.1,mlp_dim=512):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 第一层: 初始投影
        self.first_layer = Residual_Layer(in_features, out_features=mlp_dim,dropout=dropout
)
        
        # 中间层处理
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual_Layer(
                    mlp_dim, 
                    mlp_dim,  # 输入输出相同
                    dropout=dropout
                ),
                nn.Sequential(
                    nn.Linear(mlp_dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(mlp_dim) if mlp_dim > 1 else nn.Identity()
                )
            ]))
        
        # 最终输出层
        self.output_layer = nn.Linear(mlp_dim, out_features)
        self.final_activation = nn.GELU()
        
        # 初始化所有层
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化所有子模块"""
        self.first_layer.reset_parameters()
        for attn, ff in self.layers:
            attn.reset_parameters()
            # 初始化前馈网络
            for layer in ff:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        """输入形状: (batch, features)"""
        # 初始层
        x = self.first_layer(x)

        # 通过中间层
        for attn, ff in self.layers:
            # 注意力和前馈网络
            attn_output = attn(x)
            x = attn_output + x
            del attn_output  # Free memory
            
            ff_output = ff(x)
            x = ff_output + x
            del ff_output  # Free memory

        # 输出投影和激活
        output = self.output_layer(x)
        result = self.final_activation(output)
        del output, x  # Free memory
        return result
    

# 标准注意力机制
class Residual_Layer(nn.Module):

    def __init__(self, in_features, out_features=None, dropout=0.1):
        super().__init__()
        # 确保输出维度正确
        out_features = out_features or in_features
        
        # 标准线性变换层
        self.projection = nn.Linear(in_features, out_features)
    
        # 残差连接
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        # 输出处理
        self.norm = nn.LayerNorm(out_features) if out_features > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
        # 安全初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """安全的参数初始化"""
        # 初始化投影层
        nn.init.xavier_normal_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
        
        # 初始化残差连接层
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_normal_(self.residual.weight)
            if self.residual.bias is not None:
                nn.init.zeros_(self.residual.bias)

    def forward(self, x):
        """输入形状: (batch, in_features)"""
        # 保存原始输入用于残差连接
        residual = self.residual(x)
        # 标准线性变换
        output = self.projection(x)
        output = torch.nn.functional.gelu(output)
        # 残差连接和归一化
        result = self.dropout(self.norm(output + residual))
        del output  # Free memory
        return result


# 标准编码器
class Deep_Residual_Network(nn.Module):
    def __init__(self, 
                 in_features=4096,
                 out_features=500,
                 encoder_config=None):
        super().__init__()
        
        # 默认配置
        encoder_config = encoder_config or {
            'depth': 2,
            'dropout': 0.1,
            'mlp_dim': 512
        }
        
        # 创建标准编码器
        self.encoder = Deep_Residual_Layer(
            in_features=in_features,
            out_features=out_features,
            **encoder_config
        )
        
        # 直接投影路径
        self.direct_projection = nn.Linear(in_features, out_features, bias=False)
        
        # 混合比例 (初始设为0.5)
        self.mix_ratio = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        
        # 安全初始化权重
        self.reset_parameters()
    
    def reset_parameters(self):
        """安全的权重初始化"""
        # 初始化直接投影层
        nn.init.xavier_normal_(self.direct_projection.weight)
        
        # 初始化混合比例
        with torch.no_grad():
            self.mix_ratio.copy_(torch.tensor(0.5))
    
    def forward(self, x):
        """输入形状: (batch, features)"""
        # 编码器路径
        encoder_out = self.encoder(x)

        # 直接投影路径
        direct_out = self.direct_projection(x)

        # 动态混合两种输出 (使用sigmoid确保范围在0-1)
        mix_weight = torch.sigmoid(self.mix_ratio)

        return mix_weight * encoder_out + (1 - mix_weight) * direct_out

    # def forward(self, x):
    #     """输入形状: (batch, features)"""
    #     # 编码器路径
    #     # encoder_out = self.encoder(x)
        
    #     # 直接投影路径
    #     direct_out = self.direct_projection(x)
        
    #     # 动态混合两种输出 (使用sigmoid确保范围在0-1)
    #     # mix_weight = torch.sigmoid(self.mix_ratio)
        
    #     return direct_out




