
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from lib import utils,metrics
from model.pytorch.dist_teach import GTModel
from model.pytorch.dist_stu_para import Stud_M,Stud_CombM
from model.pytorch.divi_graph_3d import split_graph_3d, split_time_series, merge_time_series,visualize_subgraphs_3d,visualize_original_graph
from model.pytorch.cal_graph import _calculate_supports
import shutil
import datetime
import numpy as np
import torch
import torch.nn as nn
import math
from lib import utils
import torch.nn.functional as F
import dgl
import pickle
import matplotlib.pyplot as plt
from lib.loss_function import distillation_loss
#from model.pytorch.Quantum_GCN.py import QGCN
from thop import profile, clever_format
import yaml
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModelsDist:
    def __init__(self, models, load_pretrained,  config_filename, cuda,**kwargs):
        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
      
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        teacher_model_config_path = kwargs.get('teacher_model').get('path')
  
        with open(teacher_model_config_path,encoding='utf-8') as f:
            self._teacher_model_kwargs = yaml.load(f, Loader=yaml.FullLoader)
        self._teacher_model_kwargs = self._teacher_model_kwargs.get('model')
        self._student_model_kwargs = kwargs.get('student_model')
        self._train_kwargs = kwargs.get('train')
        self._data_kwargs['seq_len'] = int(self._student_model_kwargs.get('seq_len'))
        self._data_kwargs['horizon'] = int(self._student_model_kwargs.get('horizon'))
        #config information
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        # logging.
        self._log_dir = self._get_log_dir(models, config_filename, kwargs)
        self._writer = SummaryWriter('runs/' +self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']
        
        graph_pkl_filename = self._data_kwargs.get('graph_pkl_filename')
        if self._data_kwargs.get('use_graph'):
            _, _, adj_mx = utils.load_graph_data(graph_pkl_filename)
            self.graph = adj_mx
   
        else:
            self.graph = None
        
        self.num_nodes = int(self._student_model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._student_model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._student_model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._student_model_kwargs.get('output_dim', 1))
        self.horizon = int(self._student_model_kwargs.get('horizon', 1))  # for the decoder
        self.split = self._student_model_kwargs.get('split')
        self.use_teacher = self._student_model_kwargs.get('use_teacher')
        self.g_dim = self._student_model_kwargs.get('g_dim')
        self.add_edge = self._student_model_kwargs.get('add_edge')
        self.batch_size = int(self._student_model_kwargs.get('batch_size', 64))
        
        
                # 提取dataset_dir
        dataset_dir = kwargs['data']['dataset_dir']
        self.num_subgrids = kwargs['student_model'].get('num_subgrids')
        # 根据dataset_dir确定文件路径
        if 'metr' in dataset_dir.lower():
            file_key = 'metr'
        elif 'pems04' in dataset_dir.lower():
            file_key = 'pems04'
        elif 'pems08' in dataset_dir.lower():
            file_key = 'pems08'
        else:
            file_key = 'pems'  # 默认情况下

        config_paths = {
            'metr': f'data/dist_config/metr_graph__split_results_{self.split}_{self.num_subgrids}.pkl',
            'pems': f'data/dist_config/pems_graph__split_results_{self.split}_{self.num_subgrids}.pkl',
            'pems04': f'data/dist_config/pems04_graph__split_results_{self.split}_{self.num_subgrids}.pkl',
            'pems08': f'data/dist_config/pems08_graph__split_results_{self.split}_{self.num_subgrids}.pkl'
        }

        file_path = config_paths[file_key]

        # 调用函数保存或读取结果
        G, subG, split_indices, num_subnode, cross_edges,e_all,kr_all = self.save_or_load_split_results(file_path)
        self.student_models = []
        self.streams = [torch.cuda.Stream() for _ in range(self.num_subgrids)]  # 创建多个流
        self.split_indices = split_indices
        for i in range(self.num_subgrids):
            student_model = Stud_M(self._logger, cuda, G, e_all, kr_all, subG[i], 1,  # 单个子模型初始化
                                split_indices[i], num_subnode[i], cross_edges, **self._student_model_kwargs)
            student_model = student_model.to(self.device) if torch.cuda.is_available() else student_model
            self.student_models.append(student_model)
    
        self.student_comb_model = Stud_CombM(self._logger, cuda, G, e_all, kr_all, cross_edges, **self._student_model_kwargs)
        self.student_comb_model = self.student_comb_model.to(self.device) if torch.cuda.is_available() else self.student_comb_model
       
        input_tensor1 = torch.randn((64, self._student_model_kwargs.get('seq_len'),  int(num_subnode[0]))).to(self.device)
        input_tensor2 = torch.randn((64, self._student_model_kwargs.get('seq_len'),  self._student_model_kwargs.get('num_nodes'))).to(self.device)
        
        flops, params = profile( self.student_models[0], inputs=(input_tensor1,))
        # 格式化输出
        flops, params = clever_format([flops, params], "%.3f")
        self._logger.info(f"student_model FLOPs: {flops}, Params: {params}")
    
        flops, params = profile( self.student_comb_model, inputs=( input_tensor2,))
        # 格式化输出
        flops, params = clever_format([flops, params], "%.3f")
        self._logger.info(f"student_comb_model FLOPs: {flops}, Params: {params}")
        
        
        if self.use_teacher:
            self.teacher_model = GTModel(self._logger, self.graph, cuda, **self._teacher_model_kwargs)
            self.teacher_model = self.teacher_model.to(self.device) if torch.cuda.is_available() else self.teacher_model
        self._logger.info("config_filename:%s", config_filename)
        self._logger.info("device:%s", self.device)
        self._logger.info("Model created")
        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if load_pretrained and self.use_teacher:
            self.load_pre_model(self._data_kwargs.get('pretrained_model_dir'))
    
    
    def save_or_load_split_results(self, file_path):
        if os.path.exists(file_path):
            # 文件存在，直接读取
            with open(file_path, 'rb') as f:
                G, subG, split_indices, self.num_subgrids, num_subnode, cross_edges,e_all,kr_all = pickle.load(f)
            print("文件已存在，直接读取。")
        else:
       
            print("文件不存在，正在计算。")
            ts_all = torch.tensor(self._data['x_train'][:,:,:,0], dtype=torch.float)
            #g_ = torch.tensor(self.graph).clone().detach()
            ts_all = ts_all.clone().detach()
            G,e_all,kr_all = _calculate_supports(self.graph, pos_enc_dim = self.g_dim, pos = True)
            
            if self.split == '3D':
                subG, split_indices, cross_edges = split_graph_3d(self.graph, G, ts_all, self.num_subgrids)
                #subgraph_relation_graph = build_subgraph_relation_graph(cross_edges, subG, G)
                #visualize_subgraph_relation_graph(subgraph_relation_graph)
                color_map, subgraph_indices = visualize_subgraphs_3d(subG, cross_edges, self.num_subgrids)
                visualize_original_graph(self.graph, subgraph_indices, color_map, self.num_subgrids )
    
            num_subnode = [len(indices) for indices in split_indices]

            with open(file_path, 'wb') as f:
                pickle.dump((G, subG, split_indices, self.num_subgrids, num_subnode, cross_edges ,e_all,kr_all), f)
            print("已计算并保存结果。")


        return G, subG, split_indices, num_subnode, cross_edges, e_all,kr_all

    @staticmethod
    def _get_log_dir(loadmodel, config_name,kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            #max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            #num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            #rnn_units = kwargs['model'].get('rnn_units')
            #structure = '-'.join(
            #    ['%d' % rnn_units for _ in range(num_rnn_layers)])
            seq_len = int(kwargs['student_model'].get('seq_len'))  
            
            horizon = int(kwargs['student_model'].get('horizon')) 
            filter_type = kwargs['student_model'].get('filter_type')
            num_subgarphs = int(kwargs['student_model'].get('num_subgrids'))
            split = kwargs['student_model'].get('split')
            use_teacher = str(kwargs['student_model'].get('use_teacher'))
            #overlap = float(kwargs['student_model'].get('overlap'))
            filter_type_abbr = 'No'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = '%s_%s_l_%d_h_%d_lr_%g_bs_%d/_subG_%d_split_%s_teacher_%s' % (
                time.strftime('%Y%m%d_%H%M%S'),
                loadmodel, #filter_type_abbr, #max_diffusion_step,
                seq_len, horizon, #  structure,
                learning_rate, batch_size, num_subgarphs, split, use_teacher
                )
            base_dir = kwargs.get('base_dir')
            # base_dir = data
            log_dir = os.path.join(base_dir, 'log', loadmodel, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        shutil.copy(config_name, log_dir)
        model_py = 'model/pytorch/dist_teach.py'
        shutil.copy(model_py, log_dir)
        stu_model_py = 'model/pytorch/dist_stu_para.py'
        shutil.copy(stu_model_py, log_dir)
        dist_py = 'model/pytorch/dist_para.py'
        shutil.copy(dist_py, log_dir)
        return log_dir

    def save_model(self, epoch):
        dir = self._log_dir+'models/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        config = dict(self._kwargs)
    
        # 保存每个学生模型的状态字典
        models_state_dict = {}
        for i in range(self.num_subgrids):
            models_state_dict[f'student_model_{i}'] = self.student_models[i].state_dict()
        models_state_dict[f'student_comb_model_{i}'] = self.student_comb_model.state_dict()

        # 将所有模型的state_dict添加到config中
        config['models_state_dict'] = models_state_dict
        config['epoch'] = epoch
        
        # 保存整个配置字典
        torch.save(config, dir + '/epo%d.tar' % epoch)
        
        self._logger.info("Saved models at epoch {}".format(epoch))
        return dir + '/epo%d.tar' % epoch

    def load_pre_model(self, model_path):
        self._setup_graph()
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load(model_path, map_location='cpu')
        self.teacher_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        self._logger.info("Loaded model from {} at epoch {}".format(model_path, self._epoch_num))


    def _setup_graph(self):
        with torch.no_grad():
            self.teacher_model = self.teacher_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.teacher_model(x,self.graph)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs) #获取训练参数
        return self._train(**kwargs) #训练并返回
    
    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss and other metrics for both student_model and student_comb_model.
        :return: mean losses and metrics for both models.
        """
        with torch.no_grad():
            for model in self.student_models:
                model.eval()
            self.student_comb_model.eval()
            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()

            # 初始化损失、预测和指标的容器
            losses1, losses2 = [], []
            rmses1, rmses2 = [], []
            mapes1, mapes2 = [], []
            y_truths, y_preds1, y_preds2 = [], [], []

            for batch_idx, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                x_ = split_time_series(x, self.split_indices)
                y_ = split_time_series(y, self.split_indices)
                outputs_list = []
                h_list = []
                for i, model in enumerate(self.student_models):
                    outputs, h, _ = model(x_[i])
                    outputs_list.append(outputs)
                    h_list.append(h)
                    
                    # 计算损失
                    loss1 = self._compute_loss(y_[i], outputs)
                    rmse1, mape1 = self._compute_metrics(y_[i], outputs)
                    losses1.append(loss1.item())
                    rmses1.append(rmse1.item())
                    mapes1.append(mape1.item())
                    
                    
                sub_merge = torch.stack(h_list).to(self.device)
                sub_merge = merge_time_series(sub_merge, self.split_indices, int(self.num_nodes), self.batch_size, self.seq_len)
                student_comb_outputs, _ = self.student_comb_model(sub_merge)


                loss2 = self._compute_loss(y, student_comb_outputs)
                rmse2, mape2 = self._compute_metrics(y, student_comb_outputs)
                losses2.append(loss2.item())
                rmses2.append(rmse2.item()) 
                mapes2.append(mape2.item())
                
                y_truths.append(y.cpu())
                y_preds1.append(sub_merge.cpu())
                y_preds2.append(student_comb_outputs.cpu())

            # 计算平均损失和指标
            mean_loss1 = np.mean(losses1)
            mean_loss2 = np.mean(losses2)
            mean_rmse1 = np.mean(rmses1)
            mean_rmse2 = np.mean(rmses2)
            mean_mape1 = np.mean(mapes1)
            mean_mape2 = np.mean(mapes2)

            # 输出到 TensorBoard
            self._writer.add_scalar('{} loss student_model'.format(dataset), mean_loss1, batches_seen)
            self._writer.add_scalar('{} loss student_comb_model'.format(dataset), mean_loss2, batches_seen)

            y_truths = np.concatenate(y_truths, axis=1)
            y_preds1 = np.concatenate(y_preds1, axis=1)
            y_preds2 = np.concatenate(y_preds2, axis=1)

            # 逆缩放预测值和真实值
            y_truths_scaled = []
            y_preds_scaled1, y_preds_scaled2 = [], []

            for t in range(y_preds1.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred1 = self.standard_scaler.inverse_transform(y_preds1[t])
                y_pred2 = self.standard_scaler.inverse_transform(y_preds2[t])

                y_truths_scaled.append(y_truth)
                y_preds_scaled1.append(y_pred1)
                y_preds_scaled2.append(y_pred2)

            return (mean_loss1, mean_loss2, 
                    {'student_model': y_preds_scaled1, 'student_comb_model': y_preds_scaled2, 'truth': y_truths_scaled},
                    {"rmse1": mean_rmse1, "mape1": mean_mape1, "rmse2": mean_rmse2, "mape2": mean_mape2})


    def _train(self, base_lr, stop_patience, steps, epochs, lr_decay_ratio=0.1, log_every=1, save_model=1,
            test_every_n_epochs=10, epsilon=1.0e-5, l2_weight=1.0e-6, **kwargs):
        min_val_loss = float('inf')
        wait = 0
        loss_weight = kwargs.get('loss_weight')
        # 初始化优化器和学习率调度器


        all_student_params = []
        for sm in self.student_models:
            all_student_params.extend(sm.parameters())
        # 为每个子模型初始化优化器
  
        optimizer_s = torch.optim.Adam(all_student_params, lr=base_lr, eps=epsilon, weight_decay=l2_weight)
    
        scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=steps, gamma=lr_decay_ratio)


        # 为集成模型（student_comb_model）初始化优化器
        optimizer = torch.optim.Adam(self.student_comb_model.parameters(), lr=base_lr, eps=epsilon, weight_decay=l2_weight)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=lr_decay_ratio)


        self._logger.info('Start training ...')
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("Number of batches: {}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        train_losses1, train_losses2 = [], []
        val_losses1, val_losses2 = [], []

        for epoch_num in range(self._epoch_num, epochs):
            
            if self.use_teacher:
                self.teacher_model = self.teacher_model.eval()

            for i in range(self.num_subgrids):
                self.student_models[i] = self.student_models[i].train()
            self.student_comb_model = self.student_comb_model.train()

            train_iterator = self._data['train_loader'].get_iterator()
            losses1, losses2 = [], []
            PA = []
            PA2 = []
            dist = []
            dist2 = []

            start_time = time.time()
            for batch_idx, (x, y) in enumerate(train_iterator):
                total_loss = 0
                sub_outputs = []
                sub_h = []
                optimizer_s.zero_grad()
                optimizer.zero_grad()

                # 预处理数据并记录输入信息
                x, y = self._prepare_data(x, y)
                x_ = split_time_series(x, self.split_indices)
                y_ = split_time_series(y, self.split_indices)
                self._logger.debug(f"Batch {batch_idx} - Input X shape: {x.shape}, Input Y shape: {y.shape}")
            
                if self.use_teacher:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(x,self.graph)
                        self._logger.debug(f"Teacher model outputs shape: {teacher_outputs.shape}")
                teacher_outputs_ = split_time_series(teacher_outputs, self.split_indices)
                
                # 1. Student Model Forward Pass
                for i in range(self.num_subgrids):
                    outputs, h, pa = self.student_models[i](x_[i].to(self.device))
                    loss = self._compute_loss(outputs, y_[i])  # 根据任务自定义损失函数
                    loss = loss_weight[0]*loss + loss_weight[1]*pa
                    loss = loss + pa
                    if self.use_teacher:
                        loss_dis1 = distillation_loss(outputs, teacher_outputs_[i], y_[i])
                        loss = loss + loss_weight[2]*loss_dis1
                        dist.append(loss_dis1.item())
                   

                    # 更新每个模型的损失历史
                    losses1.append(loss.item())
                    PA.append(pa.item())
                    sub_h.append(h.detach())
                    total_loss = total_loss + loss
                total_loss.backward()
                optimizer_s.step()
                

                # 计算加权后的合并输出（集成输出）
                sub_merge = torch.stack(sub_h).to(self.device)
                sub_merge = merge_time_series(sub_merge, self.split_indices, int(self.num_nodes), self.batch_size, self.seq_len)

                # 3. Student Comb Model Forward Pass
                student_comb_outputs, pa = self.student_comb_model(sub_merge)
                self._logger.debug(f"Student Comb model outputs shape: {student_comb_outputs.shape}")
                
                # 4. 计算 Student Comb 模型损失并反向传播
                loss2 = self._compute_loss(y, student_comb_outputs)
                loss2 = loss_weight[0]*loss2 + loss_weight[1]*pa
                if self.use_teacher:
                    loss_dis2 = distillation_loss(student_comb_outputs, teacher_outputs, y)
                    loss2 = loss2 + loss_weight[2]*loss_dis2
                    dist2.append(loss_dis2.item())
                    self._logger.debug(f"Distillation loss for Student Comb model: {loss_dis2.item()}")

                loss2.backward()
                optimizer.step()

                losses2.append(loss2.item())
                PA2.append(pa.item())

            # 记录每个 epoch 的损失值
            train_loss_1 = np.mean(losses1)
            train_loss_2 = np.mean(losses2)
            train_losses1.append(train_loss_1)
            train_losses2.append(train_loss_2)

            self._logger.info(f"Epoch [{epoch_num}/{epochs}] - Train Loss1 (Student): {train_loss_1:.4f}, "
                                f"Train Loss2 (Student Comb): {train_loss_2:.4f}")
                
            # 更新学习率调度器
            scheduler_s.step()
            scheduler.step()

            print("PA:", np.mean(PA)) 
            print("PA2:", np.mean(PA2)) 
            print("dist:", np.mean(dist)) 
            print("dist2:", np.mean(dist2))



            # 验证阶段
            val_loss1, val_loss2, _, re_dci = self.evaluate(dataset='val', batches_seen=batches_seen)
            val_losses1.append(val_loss1)
            val_losses2.append(val_loss2)

            

            end_time = time.time()

            # 输出验证结果
            self._writer.add_scalar('training loss student_model', train_loss_1, epoch_num)
            self._writer.add_scalar('training loss student_comb_model', train_loss_2, epoch_num)
            self._writer.add_scalar('validation loss student_model', val_loss1, epoch_num)
            self._writer.add_scalar('validation loss student_comb_model', val_loss2, epoch_num)

            self._logger.info(f"Epoch [{epoch_num}/{epochs}] - Validation Loss1 (Student): {val_loss1:.4f}, "
                            f"Validation Loss2 (Student Comb): {val_loss2:.4f}, RMSE1: {re_dci['rmse1']:.4f}, "
                            f"MAPE1: {re_dci['mape1']:.6f}, RMSE2: {re_dci['rmse2']:.4f}, MAPE2: {re_dci['mape2']:.6f}, "
                            f"Time: {end_time - start_time:.1f}s")

            # 记录测试阶段指标
            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss1, test_loss2, _, re_dci = self.evaluate(dataset='test', batches_seen=batches_seen)
                self._logger.info(f"Epoch [{epoch_num}/{epochs}] - Test Loss1 (Student): {test_loss1:.4f}, "
                                f"Test Loss2 (Student Comb): {test_loss2:.4f}, RMSE1: {re_dci['rmse1']:.4f}, "
                                f"MAPE1: {re_dci['mape1']:.6f}, RMSE2: {re_dci['rmse2']:.4f}, MAPE2: {re_dci['mape2']:.6f}")
                
                # 保存模型
                model_file_name = self.save_model(epoch_num)
                self._logger.info(f'Model saved to {model_file_name}')

            # Early Stopping 判断逻辑
            if val_loss1 < min_val_loss:
                wait = 0
                self._logger.info(f"Val loss improved from {min_val_loss:.4f} to {val_loss1:.4f}, saving model...")
                if epoch_num > 100 and save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(f"Saved model at {epoch_num} to {model_file_name}")
                min_val_loss = val_loss1
            else:
                wait += 1
                if wait == stop_patience:
                    self._logger.warning(f"Early stopping at epoch: {epoch_num}")
                    break

        # 保存最终的损失信息
        loss_data = pd.DataFrame({
            'epoch': range(1, len(train_losses1) + 1),
            'train_loss1': train_losses1,
            'train_loss2': train_losses2,
            'val_loss1': val_losses1,
            'val_loss2': val_losses2
        })
        losspath = self._log_dir + 'loss_data.csv'
        loss_data.to_csv(losspath, index=False)
        self._logger.info(f"Loss data saved to: {losspath}")
        self.save_model(epoch_num)



    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        # 提取特定传感器的数据
        # 假设我们只关注第一个批次的数据
        # pems04 3维特征为flow, occupy, speed
       
        x0 = x[..., 0]
        y0 = y[..., 0]
        x = torch.from_numpy(x0).float()
        y = torch.from_numpy(y0).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
       
        return x, y


    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return metrics.masked_mae_torch(y_predicted, y_true)
        
    def _compute_metrics(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        rmse, mape = metrics.calculate_metrics(y_predicted, y_true)
        return rmse, mape

    
    