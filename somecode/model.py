import torch
import torch.nn as nn
import torchdiffeq as ode
import torch.nn.functional as F
from Transformer_code import *
from fusion_weight import *


def init_weight(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)


class EncodeFunc(nn.Module):
    def __init__(self, in_c, hid_c, out_c, A, dropout=0.75):
        super(EncodeFunc, self).__init__()  # 表示继承父类的所有属性和方法
        self.A = A
        self.act = nn.ReLU()  # 定义激活函数
        # 编码函数
        self.input_layer = nn.Sequential(nn.Linear(in_c, hid_c, bias=True), nn.Tanh(),
                                         nn.Linear(hid_c, hid_c, bias=True), nn.ReLU())
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, t, data):
        graph_data = GCN_ODE.process_graph(self.A)  # 变换邻接矩阵 \hat A = D_{-1/2}*A*D_{-1/2}

        flow_x = data["flow_x"]  # [B, N, H, D]
        B, N = flow_x.size(0), flow_x.size(1)  # batch_size、节点数

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D] H = 1, D = 1把最后两维缩减到一起了，这个就是把历史时间的特征放一起
        output_1 = self.input_layer(flow_x)
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX

        return output_1.unsqueeze(2)


class GCN(nn.Module):  # GCN模型，向空域的第一个图卷积
    def __init__(self, in_c, hid_c, out_c, dropout=0.75):
        super(GCN, self).__init__()  # 表示继承父类的所有属性和方法
        self.linear_1 = nn.Linear(in_c, hid_c)  # 定义一个线性层
        self.linear_2 = nn.Linear(hid_c, out_c)  # 定义一个线性层
        self.act = nn.ReLU()  # 定义激活函数
        # 编码函数
        self.input_layer = nn.Sequential(nn.Linear(1, 6, bias=True), nn.Tanh(), nn.Linear(6, 6, bias=True), nn.ReLU())
        self.output_layer = nn.Linear(1, 1, bias=True)

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N] 邻接矩阵，并且将数据送入设备

        graph_data = GCN.process_graph(graph_data)  # 变换邻接矩阵 \hat A = D_{-1/2}*A*D_{-1/2}

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)  # batch_size、节点数

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D] H = 1, D = 1把最后两维缩减到一起了，这个就是把历史时间的特征放一起

        # 纯GCN（暂定两层GCN）
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）
        output_1 = self.act(torch.matmul(graph_data, output_1))
        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C]

        # output_1 = self.input_layer(flow_x)  # 编码函数
        # print("flow's size:",flow_x.size)
        # output_1 = self.act(torch.matmul(graph_data, flow_x))  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX
        # output_1 = self.act(torch.matmul(graph_data, output_1))
        # output_1 = output_1+self.linear_1(flow_x)  # 残差连接
        # 输出层
        # output_1 = self.output_layer(output_1)
        # output_1 = self.act(output_1)
        # 第二个图卷积层
        # output_2 = self.linear_2(output_1)  # WX
        # 经过激活层后预测值全为0

        # output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C] , 就是 \hat AWX

        # output_3 = self.dropout_layer(output_1)

        return output_2.unsqueeze(2)  # 第２维的维度扩张

    @staticmethod
    def process_graph(graph_data):  # 这个就是在原始的邻接矩阵之上，再次变换，也就是\hat A = D_{-1/2}*A*D_{-1/2}
        N = graph_data.size(1)  # 获得节点的个数
        # matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
        matrix_i = torch.eye(N, dtype=torch.float)  # 定义[N, N]的单位矩阵
        graph_data += matrix_i  # [N, N]  ,就是 A+I

        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
        degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
        degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

        degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

        return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}


class GCN_ODE(nn.Module):
    def __init__(self, in_c, hid_c, out_c, A, dropout=0.75, rtol=.01, atol=.001, method='euler'):
        super(GCN_ODE, self).__init__()  # 表示继承父类的所有属性和方法
        self.A = A
        self.linear_1 = nn.Linear(in_c, hid_c)  # 定义一个线性层
        self.linear_2 = nn.Linear(hid_c, out_c)  # 定义一个线性层
        self.act = nn.ReLU()  # 定义激活函数

        # 编码函数
        # self.encode_layer = EncodeFunc(in_c, hid_c, out_c, dropout=0.75)
        self.odenet_layer = ODEblock(EncodeFunc(in_c, hid_c, out_c, A, dropout=0.75), rtol=rtol, atol=atol,
                                     method=method)
        self.output_layer = nn.Linear(6, 1, bias=True)

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, t, data):
        # output_1：（32，5886，1，6）
        output_1 = self.odenet_layer(t, data)
        # output_1 = self.encode_layer(t, data)
        # 输出层
        output_1 = self.output_layer(output_1)
        # 第二个图卷积层
        # output_2 = self.linear_2(output_1)  # WX
        # 经过激活层后预测值全为0

        # output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C] , 就是 \hat AWX

        # output_3 = self.dropout_layer(output_1)

        return output_1.unsqueeze(2)  # 第２维的维度扩张

    @staticmethod
    def process_graph(graph_data):  # 这个就是在原始的邻接矩阵之上，再次变换，也就是\hat A = D_{-1/2}*A*D_{-1/2}
        N = graph_data.size(0)  # 获得节点的个数
        matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
        graph_data += matrix_i  # [N, N]  ,就是 A+I

        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
        degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
        degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

        degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

        return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}


# 未引入静态特征
class GCN_test(nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout=0.75):
        super(GCN_test, self).__init__()  # 表示继承父类的所有属性和方法
        self.linear_1 = nn.Linear(6, hid_c)  # 定义一个线性层 2,8
        self.linear_2 = nn.Linear(hid_c, 6)  # 定义一个线性层 8,1
        self.act = nn.ReLU(True)  # 定义激活函数
        # 编码函数
        self.input_layer = nn.Sequential(nn.Linear(1, 6, bias=True), nn.Tanh(), nn.Linear(6, 6, bias=True), nn.ReLU())
        self.output_layer = nn.Linear(1, 1, bias=True)

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.adj = None

    def forward(self, data, device):
        # graph_data = GCN.process_graph(self.adj)  # 尝试对fusion weight 做归一化
        flow_x = device
        B, N = flow_x.size(0), flow_x.size(1)  # batch_size、节点数
        # (16,5886,1)
        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D] H = 1, D = 2    [16,5886,2]

        # 纯GCN（暂定两层GCN）
        output_1 = self.linear_1(flow_x)  # [16,5886,6]
        output_1 = self.act(torch.matmul(self.adj, output_1))
        output_2 = self.linear_2(output_1)  # [16,5886,1]
        output_2 = self.act(torch.matmul(self.adj, output_2))

        # output_1 = self.input_layer(flow_x)  # 编码函数
        # print("flow's size:",flow_x.size)
        # output_1 = self.act(torch.matmul(graph_data, flow_x))  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX
        # output_1 = self.act(torch.matmul(graph_data, output_1))
        # output_1 = output_1+self.linear_1(flow_x)  # 残差连接
        # 输出层
        # output_1 = self.output_layer(output_1)
        # output_1 = self.act(output_1)
        # 第二个图卷积层
        # output_2 = self.linear_2(output_1)  # WX
        # 经过激活层后预测值全为0

        # output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C] , 就是 \hat AWX

        # output_3 = self.dropout_layer(output_1)

        return output_2.unsqueeze(2)  # 第２维的维度扩张

    def process_graph(graph_data):  # 这个就是在原始的邻接矩阵之上，再次变换，也就是\hat A = D_{-1/2}*A*D_{-1/2}
        N = graph_data.size(0)  # 获得节点的个数
        # matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
        matrix_i = torch.eye(N, dtype=torch.float)  # 定义[N, N]的单位矩阵
        graph_data += matrix_i  # [N, N]  ,就是 A+I

        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
        degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
        degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

        degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

        return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}


class GCN_static(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, hidden_size3, static_feature, hid_c, out_c, fin_c,
                 in_c=2, fout_c=1, fdim_a=5886, dropout=0.0):
        super(GCN_static, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.static_feature = static_feature
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)  # 随机dropout
        self.A = None
        self.idex = None

        self.fusion = FusionWeight(fin_c, fout_c, fdim_a)
        self.wt1 = nn.Linear(hidden_size1, hidden_size3)
        self.wt2 = nn.Linear(hidden_size2, hidden_size3)
        self.gnnlayer = GCN_test(in_c, hid_c, out_c)

    def forward(self, data, device):  # x1:静态特征 N*(D-1)  x2：动态特征 N*1
        flow_x = device
        B, N = flow_x.size(0), flow_x.size(1)  # batch_size、节点数
        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D] H = 1, D = 1    [16,5886,6]

        attentionweight, graph = self.fusion(flow_x, self.idex)
        # 若无.clone().detach()计算梯度时会出错
        self.A = attentionweight.clone().detach().to(data)
        self.gnnlayer.adj = attentionweight.clone().detach().to(data)
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
            x = torch.sparse.matmul(self.A, flow_x)
            f = torch.sparse.matmul(self.A, self.static_feature)
        else:
            x = torch.matmul(self.A, flow_x)
            f = torch.matmul(self.A, self.static_feature)
        x = self.wt1(x) + self.wt2(f)
        x = self.gnnlayer(x, device)

        x = self.dropout_layer(x)
        x = F.relu(x)

        return x


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0, 1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_idex(self, idex):
        self.odefunc.idex = idex

    def forward(self, x):  # (16,5886,1,6)
        t = self.t.type_as(x)  # [0,6]
        # z1 = ode.odeint(self.odefunc, x, t, method='euler')
        z = ode.odeint(self.odefunc, x, t, method='euler')[1]  # 取出预测目标
        return z


class ODEG(nn.Module):
    def __init__(self, in_c, hid_c, out_c, adj, time=6):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(GCN_test(in_c, hid_c, out_c, adj, dropout=0.75), t=torch.tensor([0, time]))

    def forward(self, x, device):
        x = x.to(device)

        z = self.odeblock(x)
        return F.relu(z)


class ODEG_static(nn.Module):
    def __init__(self, dim_val, dim_attn, dd, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers,
                 n_heads,  # Transformer
                 hidden_size1, hidden_size2, hidden_size3, static_feature,  # GCN_static
                 fin_c, fout_c, fdim_a,  # fusion
                 in_c, hid_c, out_c, dropout=0.75, time=6):  # GCN_test
        super(ODEG_static, self).__init__()
        self.dropout = dropout
        self.encoder = nn.Sequential(nn.Linear(1, 3), nn.ReLU())
        self.decoder = nn.Linear(hidden_size1, 1)
        self.dropout_layer = nn.Dropout(dropout)
        self.transformer = Transformer(dim_val, dim_attn, dd, dec_seq_len, output_sequence_length, n_decoder_layers,
                                       n_encoder_layers, n_heads)
        self.odeblock = ODEblock(GCN_static(hidden_size1, hidden_size2, hidden_size3, static_feature,
                                            in_c, hid_c, out_c, fin_c, fout_c, fdim_a, dropout=0.75),
                                 t=torch.tensor([0, time]))

    def forward(self, x, idex, device):
        # 构造出z(t)作为初始值
        x = self.encoder(x)
        c = self.transformer(x)
        c = torch.unsqueeze(c, 2)
        input_data = torch.cat([x[:, :, 0:1, :], c], 3)  # [batchisize,N,7,6]
        self.odeblock.set_idex(idex)
        z = self.odeblock(input_data)
        z = self.decoder(z)
        z = self.dropout_layer(z)
        return F.relu(z)


'''

dd = 1
enc_seq_len = 6
dec_seq_len = 1
output_sequence_length = 1
dim_val = 10
dim_attn = 5
lr = 0.002
epochs = 20
n_heads = 3
n_decoder_layers = 3
n_encoder_layers = 3
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
x = torch.randn(size=(16, 5886, 1, 2))
graph = torch.randn(size=(5886, 5886))
static_feature = torch.randn(size=(5886, 4))
my_net = ODEG_static(dim_val, dim_attn, dd, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers,n_heads,
        hidden_size1=2, hidden_size2=4, hidden_size3=2, A=graph, static_feature=static_feature,dropout=0.75)
pre = my_net(x, device)
print(pre.size())

#测试
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
hidden_size1 = 6
hidden_size2 = 4
hidden_size3 = 2

in_c = 2
hid_c = 8
out_c = 1

fin_c = 6
fout_c = 1
fdim_a = 5886
static_feature = torch.randn(size=(5886, 4))
x = torch.randn(size=(16, 5886, 1, 6))
graph = torch.randn(size=(5886, 5886))
mynet = GCN_static(hidden_size1, hidden_size2, hidden_size3, static_feature,hid_c, out_c,
                 in_c, fin_c,fout_c,fdim_a, dropout=0.0)
'''
'''
# Transformer超参
dd = 3
enc_seq_len = 6
dec_seq_len = 1
output_sequence_length = 3
dim_val = 10
dim_attn = 5
lr = 0.002
epochs = 20
n_heads = 3
n_decoder_layers = 3
n_encoder_layers = 3
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
hidden_size1 = 6
hidden_size2 = 4
hidden_size3 = 2

in_c = 6
hid_c = 8
out_c = 6

fin_c = 6
fout_c = 1
fdim_a = 5886
static_feature = torch.randn(size=(5886, 4))
x = torch.randn(size=(16, 5886, 7, 1))
graph = torch.randn(size=(5886, 5886))
mynet = ODEG_static(dim_val, dim_attn, dd, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers,
                 n_heads,  # Transformer
                 hidden_size1, hidden_size2, hidden_size3, static_feature,  # GCN_static
                 fin_c, fout_c, fdim_a,  # fusion
                 in_c, hid_c, out_c, dropout=0.75, time=6) # GCN_test)
pre = mynet(x,0, device)
'''
