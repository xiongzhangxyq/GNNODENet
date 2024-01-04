import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch

class Evaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs(target - output) / (target + 5))  # 加５是因为target有可能为0，当然只要不太大，加几都行

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target, output)
        mape = Evaluation.mape_(target, output)
        rmse = Evaluation.rmse_(target, output)

        return mae, mape, rmse


def visualize_result(h5_file, nodes_id, time_se, visualize_file):
    file_obj = h5py.File(h5_file, "r")  # 获得文件对象，这个文件对象有两个keys："predict"和"target"
    # (5886,313)
    prediction = file_obj["predict"][:][:, :, 0]  # [N, T],切片，最后一维取第0列，所以变成二维了，要是[:, :, :1]那么维度不会缩减
    # (5886,313)
    file_obj.close()
    file_obj_1 = h5py.File(h5_file, "r")
    target = file_obj_1["target"][:][:, :, 0]  # [N, T],同上
    file_obj_1.close()

    plot_prediction = prediction[nodes_id][time_se[0]: time_se[1]]  # [T1]，将指定节点的，指定时间的数据拿出来
    plot_target = target[nodes_id][time_se[0]: time_se[1]]  # [T1]，同上

    plt.figure()
    plt.grid(True, linestyle="-.", linewidth=0.5)
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_prediction, ls="-", marker="o", color="r",
             linewidth=2, markevery=10)
    plt.plot(np.array([t for t in range(time_se[1] - time_se[0])]), plot_target, ls="-", marker="*", color="b",
             linewidth=2, markevery=10)
    plt.xlabel("Time (hour)", size=20)
    plt.ylabel("Total infected cases", size=20)
    plt.legend(["Prediction", "Target"], loc="lower right", fontsize=20)

    plt.axis([0, time_se[1] - time_se[0],
              np.min(np.array([np.min(plot_prediction), np.min(plot_target)])),
              np.max(np.array([np.max(plot_prediction), np.max(plot_target)]))])

    plt.savefig(visualize_file + "1227.png")
    plt.savefig(visualize_file + "1227.eps", format='eps', dpi=1000)



