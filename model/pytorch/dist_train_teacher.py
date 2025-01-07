
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from lib import utils,metrics

from model.pytorch.dist_teach import GTModel

import shutil
import datetime
import numpy as np
import torch
import torch.nn as nn
import math
from lib import utils
import torch.nn.functional as F
import dgl

import matplotlib.pyplot as plt



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModelsTearcher:
    def __init__(self, models, load_pretrained,  config_filename,cuda,**kwargs):
        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._teacher_model_kwargs = kwargs.get('teacher_model')
        #self._student_model_kwargs = kwargs.get('student_model')
        self._train_kwargs = kwargs.get('train')
        self._data_kwargs['seq_len'] = int(self._teacher_model_kwargs.get('seq_len'))
        self._data_kwargs['horizon'] = int(self._teacher_model_kwargs.get('horizon'))
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
            
        self.num_nodes = int(self._teacher_model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._teacher_model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._teacher_model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._teacher_model_kwargs.get('output_dim', 1))
        self.horizon = int(self._teacher_model_kwargs.get('horizon', 1))  # for the decoder
      

    
        self.tearcher_model = GTModel( self._logger,  self.graph, cuda, **self._teacher_model_kwargs)
   
        self.tearcher_model = self.tearcher_model.to(self.device) if torch.cuda.is_available() else self.tearcher_model
        self._logger.info("config_filename:%s", config_filename)
        self._logger.info("device:%s", self.device)
        self._logger.info("Model created")
        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if load_pretrained and self.use_teacher:
            self.load_pre_model(self._data_kwargs.get('pretrained_model_dir'))



    
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
            seq_len = int(kwargs['teacher_model'].get('seq_len'))  
            
            horizon = int(kwargs['teacher_model'].get('horizon')) 
            filter_type = kwargs['teacher_model'].get('filter_type')
            filter_type_abbr = 'No'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = '%s_%s_l_%d_h_%d_lr_%g_bs_%d/' % (
                time.strftime('%Y%m%d_%H%M%S'),
                loadmodel, #filter_type_abbr, #max_diffusion_step,
                seq_len, horizon, #  structure,
                learning_rate, batch_size
                )
            base_dir = kwargs.get('base_dir')
            # base_dir = data
            log_dir = os.path.join(base_dir, 'log', loadmodel, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        shutil.copy(config_name, log_dir)
        model_py = 'model/pytorch/dist_teach.py'
        shutil.copy(model_py, log_dir)
        return log_dir

    def save_model(self, epoch):
        dir = self._log_dir+'models/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        config = dict(self._kwargs)
        config['model_state_dict'] = self.tearcher_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, dir+'/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return dir+'/epo%d.tar' % epoch

    def load_pre_model(self, model_path):
        self._setup_graph()
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load(model_path, map_location='cpu')
        self.tearcher_model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        self._logger.info("Loaded model from {} at epoch {}".format(model_path, self._epoch_num))


    def _setup_graph(self):
        with torch.no_grad():
            self.tearcher_model = self.tearcher_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output= self.tearcher_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs) #获取训练参数
        return self._train(**kwargs) #训练并返回

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.tearcher_model = self.tearcher_model.eval()
            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            y_truths = []
            y_preds = []
            rmses = []
            mapes = []

            for batch_idx, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                #output,view_self_e, view_e, view_kr= self.student_model(x)
                output= self.tearcher_model(x)
                loss = self._compute_loss(y, output)
                rmse, mape = self._compute_metrics(y, output)
                losses.append(loss.item())
                mapes.append(mape.item())
                rmses.append(rmse.item())
                y_truths.append(y.cpu())
                y_preds.append(output.cpu())
                """if batch_idx % 6 == 0 and len(view_self_es) < 6:
                    view_self_es.append(view_self_e.cpu().numpy())
                    view_es.append(view_e.cpu().numpy())
                    view_krs.append(view_kr.cpu().numpy())"""
            mean_loss = np.mean(losses)
            mean_rmse = np.mean(rmses)
            mean_mape = np.mean(mapes)
            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)
            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension
            
            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)
            
            return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}, {"rmse":mean_rmse, "mape":mean_mape}




    def kd_normalize(self,logit):
        mean = logit.mean(dim=-1, keepdims=True)
        stdv = logit.std(dim=-1, keepdims=True)
        return (logit - mean) / (1e-7 + stdv)

    def dkd_loss(self,logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand):
        logits_student = self.kd_normalize(logits_student_in) if logit_stand else logits_student_in
        logits_teacher = self.kd_normalize(logits_teacher_in) if logit_stand else logits_teacher_in

        gt_mask =  self._get_gt_mask(logits_student, target)
        other_mask =  self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=-1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=-1)
        pred_student =  self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher =  self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student + 1e-7)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
            * (temperature**2)
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=-1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=-1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
            * (temperature**2)
        )
        return alpha * tckd_loss + beta * nckd_loss

 
    def _get_gt_mask(self, logits, target):
        batch_size, seq_len, num_nodes = logits.shape
        target = target.reshape(batch_size, seq_len, num_nodes).long()  # Ensure target is int64 and reshape to match logits

        # Ensure target values are within bounds [0, num_nodes)
        target = target.clamp(0, num_nodes - 1)

        mask = torch.zeros_like(logits).scatter_(2, target, 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        batch_size, seq_len, num_nodes = logits.shape
        target = target.reshape(batch_size, seq_len, num_nodes).long()  # Ensure target is int64 and reshape to match logits

        # Ensure target values are within bounds [0, num_nodes)
        target = target.clamp(0, num_nodes - 1)

        mask = torch.ones_like(logits).scatter_(2, target, 0).bool()
        return mask

    def cat_mask(self,t, mask1, mask2):
        t1 = (t * mask1).sum(dim=2, keepdims=True)
        t2 = (t * mask2).sum(dim=2, keepdims=True)
        rt = torch.cat([t1, t2], dim=2)
        return rt

    def distillation_loss(self, student_outputs, teacher_outputs, true_labels, student_en_ou, teacher_en_ou, temperature=2.0, alpha=1.0, beta=1.0, logit_stand=True):
        #print('input',student_outputs.max(),teacher_outputs.max(),student_en_ou.max(), teacher_en_ou.max())

        loss_kd2 = F.kl_div(F.log_softmax(student_en_ou / temperature, dim=-1),
                            F.softmax(teacher_en_ou / temperature, dim=-1), reduction='batchmean') * (temperature * temperature)
        
        loss_dkd = self.dkd_loss(student_outputs, teacher_outputs, true_labels, alpha, beta, temperature, logit_stand)
        #print(loss_kd2,loss_dkd)
        return  loss_kd2 + loss_dkd

    def _train(self, base_lr, stop_patience, steps, epochs, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1.0e-5, l2_weight=1.0e-6, **kwargs):

        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.tearcher_model.parameters(), lr=base_lr, eps=epsilon, weight_decay=l2_weight)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=lr_decay_ratio)
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min', factor=lr_decay_ratio, patience=lr_patience, verbose=True,min_lr= kwargs.get('min_learning_rate'))

        self._logger.info('Start training ...')
        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        train_losses=[]
        val_losses=[]
        for epoch_num in range(self._epoch_num, epochs):

            self.tearcher_model = self.tearcher_model.train()

            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            test_l = []
            iter = []
            start_time = time.time()
            iter_start_time = time.time()  
            for _, (x, y) in enumerate(train_iterator):
                
                optimizer.zero_grad()
                
                x, y = self._prepare_data(x, y)

                
                teacher_outputs = self.tearcher_model(x)
  
              
                loss = self._compute_loss(y, teacher_outputs)
               
                self._logger.debug(loss.item())
                losses.append(loss.item())

                batches_seen += 1

                loss.backward()
                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.tearcher_model.parameters(), self.max_grad_norm)
                optimizer.step()
            iter_end_time = time.time()  # 记录迭代结束时间
            iter_duration = iter_end_time - iter_start_time  # 计算迭代时间
            
            if epoch_num<10:
                iter.append(iter_duration)
                self._logger.info(f'Iteration time: {np.mean(iter):.4f} seconds')  # 记录迭代时间到日志
            train_loss=np.mean(losses)
            train_losses.append(train_loss)
            self._logger.info("epoch complete")
            self._logger.info("evaluating now!")
            val_loss, y_dic , re_dci = self.evaluate(dataset='val', batches_seen=batches_seen)
            val_losses.append(val_loss)
            #lr_scheduler.step(val_loss)
            lr_scheduler.step()
            cur_lr = optimizer.param_groups[0]['lr']
            end_time = time.time()

            self._writer.add_scalar('training loss',
                                    train_loss,
                                    batches_seen)
            
            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f},  lr: {:.8f}, val:  {:.4f}, rmse: {:.4f}, mape:{:.6f},' \
                          'time:{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           train_loss, cur_lr, val_loss, re_dci["rmse"], re_dci["mape"],
                                           #lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, y_dic , re_dci= self.evaluate(dataset='test', batches_seen=batches_seen)
                test_l.append(test_loss)
                
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f},   lr: {:.8f}, test_mae: {:.4f}, rmse: {:.4f}, mape:{:.6f},' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           train_loss, cur_lr, test_loss, re_dci["rmse"], re_dci["mape"],
                                           (end_time - start_time))
                self._logger.info(message)
                model_file_name = self.save_model(epoch_num)
                self._logger.info('saving to {}'.format(model_file_name))

            if val_loss < min_val_loss:
                wait = 0
                self._logger.info(
                            'Val loss decrease from {:.4f} to {:.4f}, '.format(min_val_loss, val_loss))
                if epoch_num > 100:
                    if save_model:
                        model_file_name = self.save_model(epoch_num)
                        self._logger.info(
                            'saving to {}'.format(model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == stop_patience:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    epochs = epoch_num
                    break
        loss_data = pd.DataFrame({
                'epoch': range(1, len(train_losses) + 1),
                'train_loss': train_losses,
                'val_loss': val_losses
            })
        losspath = self._log_dir +'loss_data.csv'
        loss_data.to_csv(losspath, index=False)
        print("损失数据已保存到:", losspath)
        

        self.save_model(epochs)
        
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
    
        #x0 =  np.concatenate((x[..., 0], x[..., -1]), axis=2)
        #y0 = np.concatenate((y[..., 0], y[..., -1]), axis=2)
        x0 = x[...,0]
        y0 = y[...,0]
        x = torch.from_numpy(x0).float()
        y = torch.from_numpy(y0).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
       
        return x, y


    def _compute_loss(self, y_true, y_predicted):

        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return metrics.masked_mae_torch(y_predicted, y_true)
        
    def _compute_metrics(self, y_true, y_predicted,):
 
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        rmse, mape = metrics.calculate_metrics(y_predicted,y_true)
        return rmse, mape
    
    