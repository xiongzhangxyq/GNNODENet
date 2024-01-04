import networkx as nx
import pickle
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()


def Bulid_OM(t):
    t = str(t)
    f = open('Data_adj/cluster_visit_list' + t + '.pkl', 'rb')
    cx = pickle.load(f)
    f.close()
    cx = cx[0]
    cx = cx.toarray()
    [abs(i) for i in cx]  # 访问矩阵中部分数值为负数
    # 将邻接矩阵聚合为较少数级
    '''
    h = 100
    last = 43
    cluster_cx = np.zeros((30, 30))
    for i in range(30):
        for j in range(30):
            cluster_cx[i][j] = sum(sum(cx[i:i + h, j:j + last]))
            if j == 29:
                cluster_cx[i][j] = sum(sum(cx[i:i + h, j:j + last]))
    '''
    cx = torch.Tensor(cx)
    cx1 = cx.T
    x1 = torch.zeros(2943, 2943)

    a = torch.cat((x1, cx), 1)
    b = torch.cat((cx1, x1), 1)
    A = torch.cat((a, b), 0)
    G = nx.from_numpy_array(A.numpy())  # 转为tensor类型
    # A_norm = A_scaler.transform(A)
    [abs(i) for i in A]  # 邻接矩阵中部分数值为负数，为保证归一化时不出错，做此处理

    return A, cx


def get_flow_data(flow_file: str) -> np.array:  # 载入数据,返回numpy的多维数组
    """
    :param flow_file: str, path of .npz file to save the traffic flow data
    :return:
        np.array(N, T, D)

    data = np.load(flow_file)

    flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]  # [N, T, D],transpose就是转置，让节点纬度在第0位，N为节点数，T为时间，D为节点特征
    # [:, :, 0]就是只取第一个特征，[:, :, np.newaxis]就是增加一个维度，因为：一般特征比一个多，即使是一个，保持这样的习惯，便于通用的处理问题
    """
    f1 = open(flow_file, 'rb')  # CBG累计感染人数
    cbg_infected_sum_all_hour = pickle.load(f1)
    f1.close()
    # cbg_infected_sum_all_hour = cbg_infected_sum_all_hour.squeeze()
    cbg_init = np.array(cbg_infected_sum_all_hour)
    A, cx = Bulid_OM(0)
    cbg_init = torch.Tensor(cbg_init).squeeze()

    cbg_init = cbg_init.numpy()
    cx = cx.numpy().T
    # Y'=A^T*X
    poi_init = np.matmul(cbg_init, cx)
    flow_data = torch.cat((torch.Tensor(cbg_init), torch.Tensor(poi_init)), 1)
    flow_data = flow_data.unsqueeze(2)
    flow_data = flow_data.permute(1, 0, 2)  # (5886,1512,1)
    flow_data = flow_data.numpy()
    return flow_data  # [N, T, D]


class LoadData(Dataset):  # 这个就是把读入的数据处理成模型需要的训练数据和测试数据，一个一个样本能读取出来
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode):
        """
        :param data_path: list, ["graph file name" , "flow data file name"], path to save the data file names.
        :param num_nodes: int, number of nodes.
        :param divide_days: list, [ days of train data, days of test data], list to divide the original data.
        :param time_interval: int, time interval between two traffic data records (mins).---5 mins
        :param history_length: int, length of history data to be used.
        :param train_mode: list, ["train", "test"].
        """
        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]
        self.test_days = divide_days[1]
        self.history_length = history_length
        self.time_interval = time_interval  # 5 min

        self.one_day_length = int(24 * 1 / self.time_interval)  # 一整天的数据量

        self.graph, self.cx = Bulid_OM(0)  # (5886,5886)
        cx = self.cx.numpy()

        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[0]),
                                                               norm_dim=1)  # self.flow_norm为归一化的基

    def __len__(self):  # 表示数据集的长度
        """
        :return: length of dataset (number of samples).
        """
        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length  # 训练的样本数　＝　训练集总长度　－　历史数据长度
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length  # 每个样本都能测试，测试样本数　＝　测试总长度
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # 取每一个样本 (x, y), index = [0, L1 - 1]这个是根据数据集的长度确定的
        """
        :param index: int, range between [0, length - 1].
        :return:
            graph: torch.tensor, [N, N].
            data_x: torch.tensor, [N, H, D].
            data_y: torch.tensor, [N, 1, D].
        """
        if self.train_mode == "train":
            index = index  # 训练集的数据是从时间０开始的，这个是每一个流量数据，要和样本（ｘ,y）区别
        elif self.train_mode == "test":
            index += self.train_days * self.one_day_length  # 有一个偏移量
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode)  # 这个就是样本（ｘ,y）

        data_x = LoadData.to_tensor(data_x)  # [N, H, D] # 转换成张量
        data_y = LoadData.to_tensor(data_y).unsqueeze(1)  # [N, 1, D]　# 转换成张量，在时间维度上扩维

        return {"graph": LoadData.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y}  # 组成词典返回
        # return {"flow_x": data_x, "flow_y": data_y}  # 组成词典返回

    @staticmethod
    def slice_data(data, history_length, index, train_mode):  # 根据历史长度,下标来划分数据样本
        """
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            data_y: np.array [N, D].
        """
        if train_mode == "train":
            start_index = index  # 开始下标就是时间下标本身，这个是闭区间
            end_index = index + history_length  # 结束下标,这个是开区间
        elif train_mode == "test":
            start_index = index - history_length  # 开始下标
            end_index = index  # 结束下标
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x = data[:, start_index: end_index]  # 在切第二维，不包括end_index
        data_y = data[:, end_index]

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):  # 预处理,归一化
        """
        :param data: np.array,原始的交通流量数据
        :param norm_dim: int,归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
        :return:
            norm_base: list, [max_data, min_data], 这个是归一化的基.
            norm_data: np.array, normalized traffic data.
        """
        norm_base = LoadData.normalize_base(data, norm_dim)  # 计算 normalize base
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)  # 归一化后的流量数据
        nn_data = np.isnan(norm_data).any()

        return norm_base, norm_data  # 返回基是为了恢复数据做准备的

    @staticmethod
    def normalize_base(data, norm_dim):  # 计算归一化的基
        """
        :param data: np.array, 原始的交通流量数据
        :param norm_dim: int, normalization dimension.归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
        :return:
            max_data: np.array
            min_data: np.array
        """
        max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D], keepdims=True就保持了纬度一致
        min_data = np.min(data, norm_dim, keepdims=True)

        return max_data, min_data  # 返回最大值和最小值

    @staticmethod
    def normalize_data(max_data, min_data, data):  # 计算归一化的流量数据，用的是最大值最小值归一化法
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original traffic data without normalization.
        :return:
            np.array, normalized traffic data.
        """
        mid = min_data
        base = max_data - min_data
        if 0 in base:
            base_test = base
            base_test[(base_test == 0)] = 1
            normalized_data = (data - mid) / base_test
        else:
            normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):  # 恢复数据时使用的，为可视化比较做准备的
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        mid = min_data
        base = max_data - min_data

        recovered_data = data * base + mid

        return recovered_data  # 原始数据

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)
