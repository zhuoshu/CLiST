import torch
import numpy as np
import argparse
import time

import csv
from torch import nn
import torch.optim as optim
from models.CLiST import CLiST

import os
import torch.nn.functional as F
import sys
from functools import partial

import logging
from logging import getLogger
import pickle
import json

import data_utils
import utils

parser = argparse.ArgumentParser()

# load configuration from files
parser.add_argument('--config_file', type=str, default='none')

# the information about the data preprocessing
parser.add_argument('--dataset_name', type=str, default='PEMS08')
parser.add_argument('--input_length', type=int, default=-1)
parser.add_argument('--predict_length', type=int, default=-1)
parser.add_argument('--scaler_type', type=str,default='zscore')
parser.add_argument('--slice_size_per_day', type=int, default=-1)

# the hyper-parameter-setting in the model
parser.add_argument('--modelid', type=str, default='none')
parser.add_argument('--hid_dim', type=int, default=-1) # d in the paper
parser.add_argument('--d_out', type=int, default=-1) # d'
parser.add_argument('--n_heads', type=int, default=-1) # h, the number of heads
parser.add_argument('--M', type=int, default=-1) # m, the number of proxy nodes
parser.add_argument('--num_layers', type=int, default=-1) # L
parser.add_argument('--addLatestX', type=int, default=1)
parser.add_argument('--hasCross', type=int, default=1)
parser.add_argument('--tcn_kernel_size', type=int, default=3, help='0:no TCN') # \tau
parser.add_argument('--hasSemb', type=int, default=1)
parser.add_argument('--hasTemb', type=int, default=1)
parser.add_argument('--spatial_dropout', type=float, default=0.1) # \phi, dropout rate
parser.add_argument('--spatial_att_dropout', type=float, default=0.1)
parser.add_argument('--st_emb_dropout', type=float, default=0.1)
parser.add_argument('--return_att', type=int, default=0)
parser.add_argument('--att_type', type=str, default='proxy')
parser.add_argument('--activation_data', type=str, default='relu')
parser.add_argument('--activation_enc', type=str, default='gelu')
parser.add_argument('--activation_dec', type=str, default='gelu')


# the hyper-parameter-setting of the training process
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float,default=0.001)
parser.add_argument('--weight_decay', type=float,default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--early_stop_step', type=int, default=20)
parser.add_argument('--lr_decay', type=int, default=1)
parser.add_argument('--huber_delta', type=int, default=2, help='delta in huber loss')
parser.add_argument('--eval_mask', type=int, default=0,
                    help='eval_mask; \{-1: not mask; 0: exclude zero; x: exclude values below x\}')
parser.add_argument('--lr_scheduler_type', type=str, default='cosinelr')
parser.add_argument('--save_output', type=int, default=0)
parser.add_argument('--note', type=str, default='CLiST')

expid = time.strftime("%m%d%H%M", time.localtime())
args = parser.parse_args()


def get_logger(log_dir, log_filename, name=None):
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    log_level = 'INFO'

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger


class trainer():
    def __init__(self, scaler,
                 in_dim, input_length, predict_length, num_nodes,
                 lrate, wdecay, device, args,
                 modelid='LiST', inverse=True):
        self._logger = getLogger()
        if modelid == 'CLiST':
            self.model = CLiST(input_length=input_length, predict_length=predict_length, num_nodes=num_nodes, in_dim=in_dim, pre_dim=in_dim,
                              hid_dim=args.hid_dim, M=args.M, tau=args.tcn_kernel_size, n_heads=args.n_heads,
                              addLatestX=bool(args.addLatestX), hasCross=bool(args.hasCross), hasTemb=bool(args.hasTemb), hasSemb=bool(args.hasSemb),
                              slice_size_per_day=args.slice_size_per_day, num_layers=args.num_layers,
                              att_type=args.att_type,
                              st_emb_dropout=args.st_emb_dropout, 
                              spatial_dropout=args.spatial_dropout, spatial_att_dropout=args.spatial_att_dropout,
                              d_out=args.d_out,
                              activation_data=args.activation_data, activation_enc=args.activation_enc, activation_dec=args.activation_dec,
                              return_att=bool(args.return_att))        
        else:
            self._logger.error(
                f'no model named `{args.modelid}`, please try again!')
            sys.exit('error')

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.model.to(device)

        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = partial(utils.huber_loss, delta=args.huber_delta)
        self.scaler = scaler
        self.clip = 1
        self.eval_mask = args.eval_mask
        self.inverse = inverse
        self.predict_length = predict_length

        self.epochs = args.epochs
        self.lr_decay = bool(args.lr_decay)
        self.lr_scheduler_type = args.lr_scheduler_type
        self.lr_decay_ratio = 0.1
        self.lr_T_max = 30
        self.lr_eta_min = 0
        self.lr_warmup_epoch = 5
        self.lr_warmup_init = 1e-6

        self.lr_scheduler = self._build_lr_scheduler()

    def _build_lr_scheduler(self):
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(
                self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'cosinelr':
                lr_scheduler = utils.CosineLRScheduler(
                    self.optimizer, t_initial=self.epochs, lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio,
                    warmup_t=self.lr_warmup_epoch, warmup_lr_init=self.lr_warmup_init)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def _forward(self, input):
        output, A = self.model(input)  # output:(B,T,N,C) #real: (B,T,N,C)
        if self.inverse:
            predict = self.scaler.inverse_transform(output)
        else:
            predict = output

        return predict

    def _metric(self, predict, real):
        if self.eval_mask == -1:
            mae = utils.mae(predict, real).item()
            rmse = utils.rmse(predict, real).item()
            mape = utils.masked_mape_torch(predict, real, 0.0).item()
        elif self.eval_mask == 0:
            mae = utils.masked_mae_torch(predict, real, 0.0).item()
            rmse = utils.masked_rmse_torch(predict, real, 0.0).item()
            mape = utils.masked_mape_torch(predict, real, 0.0).item()
        elif self.eval_mask > 0:
            mae = utils.masked_mae_torch(
                predict, real, 0.0, self.eval_mask).item()
            rmse = utils.masked_rmse_torch(
                predict, real, 0.0, self.eval_mask).item()
            mape = utils.masked_mape_torch(
                predict, real, 0.0, self.eval_mask).item()
        return mae, mape, rmse

    def train(self, input, real_val, batches_seen):  # input(B,T,N,C), real_val(B,T,N,C)
        self.model.train()
        self.optimizer.zero_grad()
        predict = self._forward(input)
        loss = self.loss(predict, real_val)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            if self.lr_scheduler_type.lower() == 'cosinelr':
                self.lr_scheduler.step_update(num_updates=batches_seen)
        mae, mape, rmse = self._metric(predict, real_val)
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        predict = self._forward(input)
        loss = self.loss(predict, real_val)
        mae, mape, rmse = self._metric(predict, real_val)
        return loss.item(), mae, mape, rmse

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

def main():
    
    expid = time.strftime("%m%d%H%M", time.localtime())
    

    device = torch.device(args.device)

    inverse = True
    
    if os.path.exists(f'./configurations/{args.config_file}'):
        with open(f'./configurations/{args.config_file}', 'r') as f:
            x = json.load(f)
            data_config = x['data_config']
            model_config = x['model_config']
            training_config = x['training_config']
            args.dataset_name =data_config['dataset_name']
            args.slice_size_per_day = data_config['slice_size_per_day']
            args.in_dim = data_config['in_dim']
            args.input_length = data_config['input_length']
            args.predict_length = data_config['predict_length']
            args.modelid = model_config['modelid']
            args.M = model_config['M']
            args.hid_dim = model_config['hid_dim']
            args.n_heads = model_config['n_heads']
            args.num_layers = model_config['num_layers']
            args.tcn_kernel_size = model_config['tcn_kernel_size']
            args.d_out = model_config['d_out']
            args.eval_mask = training_config['eval_mask']


    suffix = '' if args.note =='CLiST' else f'_{args.note}'
    args.note = f'M{args.M}_d{args.hid_dim}_h{args.n_heads}_dout{args.d_out}{suffix}'
    
    save_path = f'./experiments/{args.modelid}/{args.dataset_name}/{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp' + str(
        expid) + "/"
    log_path = save_path
    log_filename = f'{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp{expid}.log'
    log_csv_path = f'{log_path}/logTrain_{args.modelid}_{args.dataset_name}_exp'+str(expid)+'.csv'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = get_logger(log_path, log_filename)
    exp_name = f'{args.modelid}_{args.dataset_name}'
    logger.info(exp_name)

    with open(f'{save_path}/config.pkl','wb') as f:
        pickle.dump(args,f)
    with open(f'{save_path}/config.json','w') as f:
        json.dump(vars(args), f)
    
    dataloader, num_nodes, _ = data_utils.load_dataset_time(args.dataset_name, args.input_length, args.predict_length,
                                                      args.batch_size, args.batch_size, args.batch_size, scalertype=args.scaler_type,
                                                       in_dim=args.in_dim
                                                      )

    torch.cuda.empty_cache()

    scaler = dataloader['scaler']
    logger.info(args)
    logger.info(args.note)

    pre_dim = args.in_dim
    engine = trainer(scaler, args.in_dim, args.input_length, args.predict_length, num_nodes,
                     lrate=args.learning_rate, wdecay=args.weight_decay, device=device, args=args,
                     modelid=args.modelid, inverse=inverse)
    total_para_num, trainable_para_num = get_parameter_number(engine.model)    
    logger.info("start training...")
    all_start_time = time.time()
    his_loss = []
    trainloss_record = []
    val_time = []
    train_time = []
    best_validate_loss = np.inf
    validate_score_non_decrease_count = 0

    log_in_train_details = []
    batches_seen = 0
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        dataloader['train_loader'].shuffle()
        t1 = time.time()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x, xtod, xdow = x
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            trainxtod = torch.LongTensor(xtod).to(device)
            trainxdow = torch.LongTensor(xdow).to(device)
            trainx = [trainx, trainxtod, trainxdow]
            metrics = engine.train(
                trainx, trainy[..., 0:pre_dim], batches_seen)
            batches_seen += 1
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d} [{:d}], Train Loss: {:.4f},Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                logger.info(log.format(
                    iter, batches_seen, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]))
        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            x, xtod, xdow = x
            valx = torch.Tensor(x).to(device)
            valy = torch.Tensor(y).to(device)

            valxtod = torch.LongTensor(xtod).to(device)
            valxdow = torch.LongTensor(xdow).to(device)
            valx = [valx, valxtod, valxdow]

            metrics = engine.eval(valx, valy[..., 0:pre_dim])
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        logger.info(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        if engine.lr_scheduler is not None:
            if engine.lr_scheduler_type.lower() == 'reducelronplateau':
                engine.lr_scheduler.step(mvalid_loss)
            elif engine.lr_scheduler_type.lower() == 'cosinelr':
                engine.lr_scheduler.step(i)
            else:
                engine.lr_scheduler.step()

        his_loss.append(mvalid_loss)
        trainloss_record.append(
            [mtrain_loss, mtrain_mae, mtrain_mape, mtrain_mape, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse])
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} Train MAPE: {:.4f}, Train RMSE: {:.4f}, '
        log += 'Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        logger.info(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                               mvalid_rmse, (t2 - t1)))
        log_in_train_details.append([i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                                     mvalid_rmse, (t2 - t1)])

        if best_validate_loss > mvalid_loss:
            best_validate_loss = mvalid_loss
            validate_score_non_decrease_count = 0
            torch.save(engine.model.state_dict(), save_path + "best.pth")
            logger.info('got best validation result: {:.4f}, {:.4f}, {:.4f}'.format(
                mvalid_loss, mvalid_mape, mvalid_rmse))
        else:
            validate_score_non_decrease_count += 1

        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    
    avg_train_time = np.mean(train_time)
    avg_inference_time = np.mean(val_time)
    logger.info(
        "Average Training Time: {:.4f} secs/epoch".format(avg_train_time))
    logger.info("Average Inference Time: {:.4f} secs".format(avg_inference_time))
    training_time = (time.time() - all_start_time) / 60

    bestid = np.argmin(his_loss)
    logger.info("Training finished")
    logger.info(
        f"The valid loss on best model is {str(round(his_loss[bestid], 4))}")

    # testing
    engine.model.load_state_dict(torch.load(save_path + "best.pth"))
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy[:, :, :, 0:pre_dim]
    inference_time = 0
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x, xtod, xdow = x
        testx = torch.Tensor(x).to(device)
        testxtod = torch.LongTensor(xtod).to(device)
        testxdow = torch.LongTensor(xdow).to(device)
        testx = [testx, testxtod, testxdow]
        with torch.no_grad():
            t1 = time.time()
            preds, A = engine.model(testx)
            inference_time += time.time()-t1
        outputs.append(preds)
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]


    all_amae = []
    all_amape = []
    all_armse = []
    MAE_list = []
    MAPE_list = []
    RMSE_list = []
    for feature_idx in range(pre_dim):
        amae = []
        amape = []
        armse = []
        pred_feature = yhat[..., feature_idx]
        real_feature = realy[..., feature_idx]
        for i in range(args.predict_length):
            if inverse:
                pred = scaler.inverse_transform(pred_feature[:, i])
            else:
                pred = pred_feature[:, i]
            real = real_feature[:, i]

            metrics = utils.metric_torch(pred, real, mask_val=args.eval_mask)
            log = 'Evaluate best model on test data for [dim{:d}] horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            logger.info(log.format(feature_idx, i + 1,
                        metrics[0], metrics[1], metrics[2]))
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])

        log = '[dim{:d}] On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        MAE = np.mean(amae)
        MAPE = np.mean(amape)
        RMSE = np.mean(armse)
        logger.info(log.format(
            feature_idx, args.predict_length, MAE, MAPE, RMSE))
        MAE_list.append(MAE)
        MAPE_list.append(MAPE)
        RMSE_list.append(RMSE)
        all_amae.append(amae)
        all_amape.append(amape)
        all_armse.append(armse)

    if not os.path.exists(f'./results/{args.modelid}/'):
        os.makedirs(f'./results/{args.modelid}/')
    result_txt_path = f'./results/{args.modelid}/{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_Results.txt'
    result_csv_path = f'./results/{args.modelid}/{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_Results.csv'
    train_metric = trainloss_record[bestid][0:4]
    valid_metric = trainloss_record[bestid][4:]

    with open(result_csv_path, 'a+', newline='')as f0:
        f_csv = csv.writer(f0)
        row = [expid, args.dataset_name, args.note,args.M,args.hid_dim,args.n_heads,'test']
        for feature_idx in range(pre_dim):
            row.extend([MAE_list[feature_idx], MAPE_list[feature_idx]
                       * 100, RMSE_list[feature_idx]])
        row.extend(['val'])
        row.extend(valid_metric)
        row.extend(['train'])
        row.extend(train_metric)
        f_csv.writerow(row)

    with open(log_csv_path, 'a+', newline='')as flog:
        flog_csv = csv.writer(flog)
        flog_csv.writerow(['epoch', 'train_loss', 'train_mae', 'train_mape', 'train_rmse', 
                           'valid_loss', 'valid_mae', 'valid_mape', 'valid_rmse', 'train_time'])
        for it in log_in_train_details:
            flog_csv.writerow(it)

    with open(result_txt_path, 'a+') as f:
        f.write(
            f"\n【{expid}】{args.note} {args.dataset_name} epoch={len(trainloss_record)} bestid={bestid}:")
        f.write(f'\n{args} in-dim={args.in_dim}')
        f.write(
            f'\ntotal_para_num={total_para_num}, trainable_para_num={trainable_para_num}')
        f.write(
            f'\ntraining_time={training_time}min, inference_time={inference_time}s')
        f.write(
            f'\navg_training_time={avg_train_time}s/epoch, avg_inference_time={avg_inference_time}s')
        f.write('\ntrain metric:')
        for id, it in enumerate(train_metric):
            f.write('%.4f\t' % it)
        f.write('\nvalid metric:')
        for id, it in enumerate(valid_metric):
            f.write('%.4f\t' % it)
        for feature_idx in range(pre_dim):
            f.write(f'\nfeature {feature_idx}:')
            f.write('\nMAE_list:')
            for id, it in enumerate(all_amae[feature_idx]):
                f.write('%.4f\t' % it)
            f.write('\nMAPE_list:')
            for id, it in enumerate(all_amape[feature_idx]):
                f.write('%.4f\t' % it)
            f.write('\nRMSE_list:')
            for id, it in enumerate(all_armse[feature_idx]):
                f.write('%.4f\t' % it)

            f.write('\nOn average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(MAE_list[feature_idx],
                                                                                                                       MAPE_list[feature_idx],
                                                                                                                       RMSE_list[feature_idx]))
            f.flush()
    logger.info('log and results saved.')
    if bool(args.save_output):
        output_path = f'{save_path}/{args.modelid}_{args.dataset_name}_exp{expid}_output.npz'
        np.savez_compressed(output_path, prediction = yhat.cpu().numpy(), truth=realy.cpu().numpy())
        logger.info('output npz saved.')

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
