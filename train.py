
import numpy as np
import random
import torch

from options.base_options import BaseOptions
from data import create_dataset
from util.visualizer import Visualizer
from envclass import envclass
from models.ours_model import MBPNModel
import time

# 设置随机数种子
seed = 12345
random.seed(seed)  # Python内置随机库
np.random.seed(seed)  # NumPy
torch.manual_seed(seed)  # PyTorch
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # 如果使用了CUDA，还需要为所有GPU设置种子
    
if __name__ == '__main__':
    opt = BaseOptions().parse()  # 获取训练选项
    dataset = create_dataset(opt)  # 导入数据集以生成数据集路径集合
    model = MBPNModel(opt)  # 创建模型
    model.setup(opt)  # 设置模型，例如学习率
    visualizer = Visualizer(opt)  # 显示和保存打印的图像
    total_iters = 0  # 总迭代次数

    myenv = envclass(opt)  # 测试模型索引
    total_start_time = time.time()

    for epoch in range(opt.epoch_count, opt.maxepoch + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0  # 本epoch的迭代次数

        for i, data in enumerate(dataset):
            iter_start_time = time.time()

            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), total_iters)

            if total_iters % opt.print_freq == 0:
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, model.get_current_losses(), t_comp)
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

            savebest = myenv.env(epoch)
            if savebest == "bestmode":
                model.save_networks('bestmode')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.maxepoch, time.time() - epoch_start_time))
        model.update_learning_rate()

    total_finish_time = (time.time() - total_start_time)
    visualizer.log_total_time(total_finish_time)
    print('Total training time: {:.1f} hours'.format(total_finish_time / 3600))
    print('End of train epoch %d ' % epoch)

