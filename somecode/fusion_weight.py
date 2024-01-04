import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Bulid_OM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_graph(graph_data):  # 这个就是在原始的邻接矩阵之上，再次变换，也就是\hat A = D_{-1/2}*A*D_{-1/2}
    N = graph_data.size(1)  # 获得节点的个数
    # matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
    matrix_i = torch.eye(N, dtype=torch.float).to(device)  # 定义[N, N]的单位矩阵
    graph_data += matrix_i  # [N, N]  ,就是 A+I

    degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
    degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
    degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

    degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

    return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.F = F.softmax
        self.W = nn.Linear(in_c, out_c, bias=False)  # y = W * x  w:1*1
        self.b = nn.Parameter(torch.Tensor(out_c))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        """
        :param inputs: input features, [B, N, C].
        :param graph: graph structure, [N, N].
        :return:
            output features, [B, N, D].
        """
        h = self.W(inputs)  # [B, N, D]，一个线性层，就是第一步中公式的 W*h


        # 下面这个就是，第i个节点和第j个节点之间的特征做了一个内积，表示它们特征之间的关联强度
        # 再用graph也就是邻接矩阵相乘，因为邻接矩阵用0-1表示，0就表示两个节点之间没有边相连
        # 那么最终结果中的0就表示节点之间没有边相连
        b = graph.unsqueeze(0)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        outputs = torch.bmm(h, h.transpose(1, 2)) * graph  # [B, N, D]*[B, D, N]->[B, N, N],         x(i)^T * x(j)

        # 由于上面计算的结果中0表示节点之间没关系，所以将这些0换成负无穷大，因为softmax的负无穷大=0
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))

        attention = self.F(outputs, dim=1)  # [B, N, N]，在第２维做归一化，就是说所有有边相连的节点做一个归一化，得到了注意力系数
        return attention


class FusionWeight(torch.nn.Module):
    def __init__(self, in_c, out_c, dim_a):
        super(FusionWeight, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.dim_va = dim_a
        self.graphAttentionLayer = GraphAttentionLayer(in_c, out_c)
        self.fc1 = nn.Linear(dim_a, dim_a, bias=False)

    def forward(self, input, t_stat):
        adj, cx = Bulid_OM(t_stat)
        input = input.to(device)
        adj = adj.to(device)
        attention = self.graphAttentionLayer(input,adj)
        adj = process_graph(adj)  # 归一化
        adj2 = self.fc1(adj)  # 邻接矩阵做线性变换
        return adj2 + attention[0], adj


'''
fusion = FusionWeight(1, 1, 5886)
x = torch.randn(size=(16, 5886, 1))
graph = torch.randn(size=(5886, 5886))
attentionweight,adj = fusion(x, 0)
print("hello")
'''
