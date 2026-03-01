import torch
import torch.nn as nn

class DeepHydroMLP(nn.Module):
    def __init__(self, input_dim=12, output_dim=6):
        super(DeepHydroMLP, self).__init__()
        
        # 定义神经网络结构
        # 结构设计：输入层(12) -> 宽隐层(128) -> 更宽隐层(256) -> 宽隐层(128) -> 输出层(6)
        # 这种 "Bottleneck" (瓶颈) 结构适合拟合物理规律
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),  # LeakyReLU 保留一点负梯度，防止神经元坏死
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            
            nn.Linear(128, output_dim)
        )

        # 权重初始化：告诉网络一开始怎么“猜”比较靠谱
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                # Kaiming 初始化：专为 ReLU/LeakyReLU 设计的初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 前向传播：数据流过网络
        return self.net(x)