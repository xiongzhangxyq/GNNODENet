import os
import time
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torchdiffeq as ode
from torch.optim.lr_scheduler import StepLR
from dataset import LoadData, Bulid_OM
from utils import Evaluation
from utils import visualize_result
from gcnnet import *
from Transformer_code import *
import warnings
from fusion_weight import *
import pandas as pd
from gat import GATNet

warnings.filterwarnings('ignore')
torch.random.manual_seed(420)
ss = StandardScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 静态特征及其归一化
static_feature = []
cbg_df = pd.read_csv('Data_features/cbg_size_san.csv')
poi_time_feet = pd.read_csv('Data_features/average_feet_time_all.csv')
for i, row in cbg_df.iterrows():
    static_feature.append([1, row[1], 0, 0])  # 1代表cbg，0代表poi
for i, row in poi_time_feet.iterrows():
    static_feature.append([0, 0, row[1], row[2]])
static_feature = np.array(static_feature)
static_scaler = ss.fit_transform(static_feature)
static_feature = torch.Tensor(static_feature)
static_feature = static_feature.to(device)

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

# GCN部分超参
hidden_size1 = 6
hidden_size2 = 4
hidden_size3 = 2

in_c = 6
hid_c = 8
out_c = 6

# 编解码部分超参

# fusion weight 超参
fin_c = 6
fout_c = 1
fdim_a = 5886

TallN = 5886
CBGN = 2943


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 配置GPU,因为可能有多个GPU，这里用了第0号GPU
    train_data = LoadData(data_path=["Data_features/cbg.pkl", "Data_features/cbg_omicro.pkl"], num_nodes=TallN,
                          divide_days=[42, 21],
                          time_interval=1, history_length=7, train_mode="train")
    train_loader = DataLoader(train_data, batch_size, shuffle=False)
    test_data = LoadData(data_path=["Data_features/cbg.pkl", "Data_features/cbg_omicro.pkl"], num_nodes=TallN,
                         divide_days=[42, 21],
                         time_interval=1, history_length=7, train_mode="test")
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    # my_net = ODEG(in_c=2, hid_c=8, out_c=1, adj=graph, time=6)  # 无 static features
    # my_net = GCN_test(in_c=1, hid_c=6, out_c=1, adj=graph, dropout=0.75)  # 加载GCN模型
    # my_net = GCN_ODE(in_c=1, hid_c=6, out_c=1, dropout=0.75)  # 加载GCN_ODE模型
    # my_net = ChebNet(in_c=1, hid_c=6, out_c=1, K=2)         # 加载ChebNet模型
    # my_net = GATNet(in_c=2, hid_c=6, out_c=1, n_heads=2)    # 加载GAT模型
    # my_net = ODEG_static(dim_val, dim_attn, dd, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers,
    #                     n_heads,
    #                     hidden_size1, hidden_size2, hidden_size3, graph, static_feature,
    #                     in_c, hid_c, out_c, dropout=0.75, time=6)  # 有static features
    my_net = ODEG_static(dim_val, dim_attn, dd, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers,
                         n_heads,  # Transformer
                         hidden_size1, hidden_size2, hidden_size3, static_feature,  # GCN_static
                         fin_c, fout_c, fdim_a,  # fusion
                         in_c, hid_c, out_c, dropout=0.75, time=6)  # GCN_test)
    # my_net.apply(init_weight)  # 初始化权重和偏置
    my_net = my_net.to(device)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    optimizer = optim.Adam(params=my_net.parameters(), lr=0.005)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    Epoch = 400  # 训练的次数
    my_net.train()  # 打开训练模式
    Train_loss = []
    for epoch in range(Epoch):
        epoch_loss = 0.0

        start_time = time.time()
        acount = 0
        for idex, data in enumerate(
                train_loader):  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]],一次把一个batch的训练数据取出来
            input = data["flow_x"].to(device)
            print("训练:",idex)
            target = data["flow_y"].to(device)
            # attentionweight, graph = fusion(input.squeeze(2), idex)
            with torch.autograd.set_detect_anomaly(True):
                my_net.zero_grad()  # 梯度清零
                if idex == 0:
                    predict_value = my_net(input, idex, device)  # [16,5886,1,2]
                else:
                    predict_value = my_net(input, idex, device)  # [16,5886,1,2]

                # 尝试引入L1正则项
                re_loss = 0
                lam = 0.1
                # in_loss = criterion(predict_value[:, 0:CBGN, :, 0:1], target[:, 0:CBGN, :, :])  # 计算损失，切记这个loss不是标量
                # for name, param in my_net.named_parameters():
                #     if param.requires_grad:
                #         re_loss += torch.sum(torch.abs(param))
                # loss = in_loss + lam * re_loss
                loss = criterion(predict_value[:, 0:CBGN, :, 0:1], target[:, 0:CBGN, :, :])
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()  # 更新参数
        end_time = time.time()
        b = 1000 * epoch_loss / len(train_data)
        Train_loss.append(b)
        #print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
        #                                                                  (end_time - start_time) / 60))
    with open("./train_loss.txt", 'w') as train_los:
        train_los.write(str(Train_loss))

    # Test Model
    my_net.eval()
    with torch.no_grad():
        MAE, MAPE, RMSE = [], [], []
        Target1 = np.zeros([CBGN, 1, 1])  # [N, T, D],T=1 ＃ 目标数据的维度，用０填充
        Predict1 = np.zeros_like(Target1)  # [N, T, D],T=1 # 预测数据的维度

        Target2 = np.zeros([CBGN, 1, 1])  # [N, T, D],T=1 ＃ 目标数据的维度，用０填充
        Predict2 = np.zeros_like(Target2)  # [N, T, D],T=1 # 预测数据的维度

        total_loss = 0.0
        Test_loss = []
        for idex, data in enumerate(test_loader):
            print("测试：",idex)
            # 下面得到的预测结果是归一化的结果
            input = data["flow_x"].to(device)
            target = data["flow_y"].to(device)
            # attentionweight, graph = fusion(input.squeeze(2), idex)
            my_net.zero_grad()  # 梯度清零
            predict_value = my_net(input, idex, device)
            loss = criterion(predict_value[:, 0:CBGN, :, 0:1], target[:, 0:CBGN, :, :])  # 使用MSE计算loss
            Test_loss.append(loss.item())
            total_loss += loss.item()
            # 下面把预测值和目标值的batch放到第二维的时间维度，这是因为在测试数据的时候对样本没有shuffle，
            # 所以每一个batch取出来的数据就是按时间顺序来的，因此放到第二维来表示时间是合理的.
            predict_value1 = predict_value[:, 0:CBGN, :, 0:1].transpose(0, 2).squeeze(0)  # [2943,16,1]
            target_value1 = target[:, 0:CBGN, :, :].transpose(0, 2).squeeze(0)  # [2943,16,1]

            performance1, data_to_save1 = compute_performance(predict_value1, target_value1,
                                                              test_loader)  # 计算模型的性能，返回评价结果和恢复好的数据

            # 下面这个是每一个batch取出的数据，按batch这个维度进行串联，最后就得到了整个时间的数据，也就是
            # [N, T, D] = [N, T1+T2+..., D]
            Predict1 = np.concatenate([Predict1, data_to_save1[0]], axis=1)
            Target1 = np.concatenate([Target1, data_to_save1[1]], axis=1)

            MAE.append(performance1[0])
            MAPE.append(performance1[1])
            RMSE.append(performance1[2])

            print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))
        torch.save(my_net.state_dict(), 'ours(1227).pt')

    with open("./train_loss.txt", 'w') as test_loss:
        test_loss.write(str(Test_loss))
    # 三种指标取平均
    print("Performance:  MAE {:2.2f}    MAPE {:2.2f}%    RMSE{:2.2f}".format(np.mean(MAE), np.mean(MAPE * 100),
                                                                             np.mean(RMSE)))

    result_file = "ours(1227).h5"
    file_obj = h5py.File(result_file, "w")
    file_obj["predict"] = Predict1  # [N, T, D]
    file_obj["target"] = Target1  # [N, T, D]

    result_file = "Ct.h5"
    file_obj = h5py.File(result_file, "w")
    file_obj["predict"] = Predict2  # [N, T, D]
    file_obj["target"] = Target2  # [N, T, D]


def compute_performance(prediction, target, data):  # 计算模型性能
    # 下面的try和except实际上在做这样一件事：当训练+测试模型的时候，数据肯定是经过dataloader的，所以直接赋值就可以了
    # 但是如果将训练好的模型保存下来，然后测试，那么数据就没有经过dataloader，是dataloader型的，需要转换成dataset型。
    try:
        dataset = data.dataset
    except:
        dataset = data

    # 对预测和目标数据进行逆归一化,flow_norm为归一化的基，flow_norm[0]为最大值，flow_norm[1]为最小值
    print("prediction：", prediction.size())

    prediction = LoadData.recover_data(dataset.flow_norm[0][0:CBGN, :, :], dataset.flow_norm[1][0:CBGN, :, :],
                                       prediction.data.cpu().numpy())
    target = LoadData.recover_data(dataset.flow_norm[0][0:CBGN, :, :], dataset.flow_norm[1][0:CBGN, :, :],
                                   target.data.cpu().numpy())  # [5886,16,1]

    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))  # 变成常向量才能计算这三种指标

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data  # 返回评价结果，以及恢复好的数据（为可视化准备的）




print("nihao")
if __name__ == '__main__':
    main()
    # 可视化
    # visualize_result(h5_file="D:\PycharmProjects\GNN-ODEnet-prediction\Figures\GCN_ODEnet_result.h5", nodes_id=2160, time_se=[100, 313])#
    # visualize_result(h5_file="ODEG_static_result.h5", nodes_id=120, time_se=[100, 313])#
