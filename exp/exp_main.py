from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import stom, Linear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.scientific_report import plot_aligned_heatmap, mts_visualize, mts_visualize_horizontal

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from collections import OrderedDict

import os
import time
import warnings
import matplotlib.pyplot as plt
import logging
import copy
import csv
import json
import wandb

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

torch._logging.set_logs(dynamo=logging.INFO)
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.suppress_errors = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def create_directory(path):
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        # Initialize WandB
        project_name = getattr(self.args, 'project_name', 'TS_Forecasting')

        if wandb.run is None:
            wandb.init(
                project=project_name,
                name=f"{self.args.model_id}_{time.strftime('%Y%m%d-%H%M%S', time.localtime())}",
                config=self.args,
                reinit=True
            )

        self.vali_times = 0
        self.test_times = 0
        self.steps = 0
        self.test_every = self.args.test_every
        self.early_stop = False
        self.specmix_writer = None
        if (self.args.predictor == 'SpecMix' or getattr(self.args, 'use_stom', 0)) and getattr(self.args,
                                                                                               'specmix_tensorboard_dir',
                                                                                               '') and SummaryWriter is not None:
            self.specmix_writer = SummaryWriter(self.args.specmix_tensorboard_dir)

    def _unwrap_model(self):
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        return model

    def _set_specmix_diagnostics(self, enabled):
        model = self._unwrap_model()
        if getattr(self.args, 'use_stom', 0) and hasattr(model, 'set_stom_diagnostics'):
            model.set_stom_diagnostics(enabled)
        if self.args.predictor == 'SpecMix' and hasattr(model, 'set_specmix_diagnostics'):
            model.set_specmix_diagnostics(enabled)

    def _get_specmix_diagnostics(self, reset=True):
        model = self._unwrap_model()
        if getattr(self.args, 'use_stom', 0) and hasattr(model, 'get_stom_diagnostics'):
            return model.get_stom_diagnostics(reset=reset)
        if self.args.predictor == 'SpecMix' and hasattr(model, 'get_specmix_diagnostics'):
            return model.get_specmix_diagnostics(reset=reset)
        return None

    def _ensure_specmix_curve(self):
        path = getattr(self.args, 'specmix_training_curve_path', '')
        if not (self.args.predictor == 'SpecMix' or getattr(self.args, 'use_stom', 0)) or not path:
            return None
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(path):
            diagnostic_fields = [
                'use_stom', 'operator', 'ma_kernel', 'local_kernel', 'rank',
                'n_spec_heads', 'gate_hidden_dim', 'gate_prior', 'gate_shape',
                'gate_phase_mean', 'gate_spec_mean', 'gate_smooth_mean',
                'gate_local_mean', 'gate_entropy',
                'gate_phase_by_horizon', 'gate_spec_by_horizon',
                'gate_smooth_by_horizon', 'gate_local_by_horizon',
                'gate_entropy_by_horizon',
                'y_phase_norm', 'y_spec_norm', 'y_smooth_norm', 'y_local_norm',
                'contribution_phase_abs', 'contribution_spec_abs',
                'contribution_smooth_abs', 'contribution_local_abs',
                'contribution_ratio_phase', 'contribution_ratio_spec',
                'contribution_ratio_smooth', 'contribution_ratio_local',
                'A_entropy', 'A_entropy_per_head', 'EML', 'EML_per_head',
                'same_phase_mass', 'same_phase_mass_per_head',
                'recent_mass', 'recent_mass_per_head', 'pressure_logZ_mean',
                'alpha_h', 'phase_strength_h', 'recent_strength_h',
                'sigma_phase_h', 'sigma_recent_h', 'spec_head_mix', 'spec_gain',
            ]
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'val_loss', 'test_loss', 'lr',
                    *diagnostic_fields,
                ])
        return path

    def _write_specmix_epoch_log(self, epoch, train_loss, val_loss, test_loss, lr, diag):
        if not (self.args.predictor == 'SpecMix' or getattr(self.args, 'use_stom', 0)) or diag is None:
            return

        diagnostic_fields = [
            'use_stom', 'operator', 'ma_kernel', 'local_kernel', 'rank',
            'n_spec_heads', 'gate_hidden_dim', 'gate_prior', 'gate_shape',
            'gate_phase_mean', 'gate_spec_mean', 'gate_smooth_mean',
            'gate_local_mean', 'gate_entropy',
            'gate_phase_by_horizon', 'gate_spec_by_horizon',
            'gate_smooth_by_horizon', 'gate_local_by_horizon',
            'gate_entropy_by_horizon',
            'y_phase_norm', 'y_spec_norm', 'y_smooth_norm', 'y_local_norm',
            'contribution_phase_abs', 'contribution_spec_abs',
            'contribution_smooth_abs', 'contribution_local_abs',
            'contribution_ratio_phase', 'contribution_ratio_spec',
            'contribution_ratio_smooth', 'contribution_ratio_local',
            'A_entropy', 'A_entropy_per_head', 'EML', 'EML_per_head',
            'same_phase_mass', 'same_phase_mass_per_head',
            'recent_mass', 'recent_mass_per_head', 'pressure_logZ_mean',
            'alpha_h', 'phase_strength_h', 'recent_strength_h',
            'sigma_phase_h', 'sigma_recent_h', 'spec_head_mix', 'spec_gain',
        ]

        path = self._ensure_specmix_curve()
        row = [
            epoch, train_loss, val_loss, test_loss, lr,
            *[diag.get(name) for name in diagnostic_fields],
        ]
        if path:
            with open(path, 'a', newline='') as f:
                csv.writer(f).writerow(row)

        if self.specmix_writer is not None:
            base_scalars = {
                'train/loss': train_loss,
                'val/loss': val_loss,
                'test/loss': test_loss,
                'learning_rate': lr,
            }
            diagnostic_scalars = [
                'gate_phase_mean', 'gate_spec_mean', 'gate_smooth_mean',
                'gate_local_mean', 'gate_entropy',
                'y_phase_norm', 'y_spec_norm', 'y_smooth_norm', 'y_local_norm',
                'contribution_phase_abs', 'contribution_spec_abs',
                'contribution_smooth_abs', 'contribution_local_abs',
                'contribution_ratio_phase', 'contribution_ratio_spec',
                'contribution_ratio_smooth', 'contribution_ratio_local',
                'A_entropy', 'EML', 'same_phase_mass', 'recent_mass',
                'pressure_logZ_mean', 'spec_gain',
            ]
            for key, value in base_scalars.items():
                if value is not None:
                    self.specmix_writer.add_scalar(key, float(value), epoch)
            for name in diagnostic_scalars:
                value = diag.get(name)
                if value is not None:
                    self.specmix_writer.add_scalar(f'stom/{name}', float(value), epoch)

            diagnostic_series = [
                'gate_phase_by_horizon', 'gate_spec_by_horizon',
                'gate_smooth_by_horizon', 'gate_local_by_horizon',
                'gate_entropy_by_horizon',
                'A_entropy_per_head', 'EML_per_head',
                'same_phase_mass_per_head', 'recent_mass_per_head',
                'alpha_h', 'phase_strength_h', 'recent_strength_h',
                'sigma_phase_h', 'sigma_recent_h', 'spec_head_mix',
            ]
            for name in diagnostic_series:
                for i, value in enumerate(diag.get(name, [])):
                    self.specmix_writer.add_scalar(f'stom/{name}_{i}', float(value), epoch)
            self.specmix_writer.flush()

    def _write_specmix_final_diagnostics(self, setting, diag):
        path = getattr(self.args, 'specmix_diagnostics_path', '')
        if not (self.args.predictor == 'SpecMix' or getattr(self.args, 'use_stom', 0)) or not path or diag is None:
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            'setting': setting,
            'model_id': self.args.model_id,
            'dataset': self.args.data,
            'pred_len': self.args.pred_len,
            'predictor': self.args.predictor,
            'use_stom': int(getattr(self.args, 'use_stom', 0)),
            **diag,
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

    def _build_model(self):
        model_dict = {
            'STOM': stom,
            'Linear': Linear,
        }
        model = model_dict[self.args.model].Model(self.args)

        if self.args.resume != 'none':
            state_dict = torch.load(self.args.resume, map_location='cpu')

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict, strict=False)

        if self.args.compile:
            model = torch.compile(model)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        params = self.model.parameters()
        model_optim = optim.AdamW(params, lr=self.args.learning_rate, betas=(0.9, 0.999),
                                  weight_decay=self.args.weight_decay)
        # model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, label='vali'):
        total_loss = []
        preds = []
        preds_add = []
        trues = []
        self.model.eval()
        print(f'Start Validation ({label})')
        with torch.no_grad():
            for i, data in enumerate(vali_loader):
                batch_x = data[0].float().to(self.device, non_blocking=True)
                batch_y = data[1].float().to(self.device, non_blocking=True)
                batch_x_mark = data[2].float().to(self.device, non_blocking=True)
                batch_y_mark = data[3].float().to(self.device, non_blocking=True)

                pred_len = self.args.pred_len  # batch_y.shape[1] - batch_x.shape[1]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device,
                                                                                                      non_blocking=True)
                # encoder - decoder
                self._set_specmix_diagnostics(label == 'test' and i == 0)
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                additional_loss = 0
                f_dim = -self.args.number_of_targets
                if self.args.features == 'MS':
                    f_dim = -1
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred.float().numpy(force=True))
                trues.append(true.float().numpy(force=True))
                loss = criterion(pred, true)

                total_loss.append(loss)

        print(f'Validation ({label}): Inference Finished')

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        mae_ot, mse_ot, rmse_ot, mape_ot, mspe_ot, rse_ot, corr_ot = metric(preds[:, :, -1], trues[:, :, -1])

        metrics_dict = {'MSE': mse, 'MAE': mae, 'MAPE': mape, 'MSPE': mspe}

        total_loss = mse

        if label == 'test':
            print(f'Validation ({label}): Visualization')

            wandb.log({
                f'Loss/{label}LossAvg': float(total_loss),
                f'Loss/{label}LossMSEAvg': float(mse),
                f'Loss/{label}LossMAEAvg': float(mae),
                f'Loss/{label}LossRMSEAvg': float(rmse),
                f'Loss/{label}OTLossMSEAvg': float(mse_ot),
                f'Loss/{label}OTLossMAEAvg': float(mae_ot),
                f'Loss/{label}OTLossRMSEAvg': float(rmse_ot),
                f'Loss/{label}LossMAPEAvg': float(mape),
                f'Loss/{label}LossMSPEAvg': float(mspe),
                'test_times': self.test_times,
                'step': self.steps
            })

            pred = pred.float().numpy()
            cbatch_x = torch.cat([batch_x[:, :, f_dim:], batch_y], dim=1).detach().cpu()
            cbatch_x = cbatch_x.float().numpy()
            met = f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, MSPE: {mspe:.4f}'

            if self.test_times % self.args.plot_every == 0:
                fig = mts_visualize(pred[0, :, -256:], cbatch_x[0, :, -256:], split_step=batch_x.shape[1], title=met,
                                    dpi=72, col_names=vali_data.col_names)
                if not os.path.exists("imgs"): os.makedirs("imgs")
                if not os.path.exists(f"imgs/{self.args.model_id}"): os.makedirs(f"imgs/{self.args.model_id}")
                fig.savefig(f"imgs/{self.args.model_id}/{self.test_times}.pdf", format="pdf", bbox_inches='tight')

                wandb.log({'MTS_VS': wandb.Image(fig), 'step': self.steps})

                plt.clf()
                if not os.path.exists("imgs_testset"): os.makedirs("imgs_testset")
                if not os.path.exists(f"imgs_testset/{self.args.model_id}"): os.makedirs(
                    f"imgs_testset/{self.args.model_id}")

            self.test_times += 1

        if label == 'vali':
            wandb.log({
                f'Loss/{label}LossAvg': float(total_loss),
                f'Loss/{label}LossMSEAvg': float(mse),
                f'Loss/{label}LossMAEAvg': float(mae),
                f'Loss/{label}LossRMSEAvg': float(rmse),
                f'Loss/{label}LossMAPEAvg': float(mape),
                f'Loss/{label}LossMSPEAvg': float(mspe),
                'vali_times': self.vali_times,
                'step': self.steps
            })
            self.vali_times += 1

        diag = None
        if (self.args.predictor == 'SpecMix' or getattr(self.args, 'use_stom', 0)) and label == 'test':
            diag = self._get_specmix_diagnostics(reset=True)
        self._set_specmix_diagnostics(False)
        self.model.train()
        print('Validation Finished')

        metrics_dict['SpecMixDiag'] = diag
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        disable_train_test_eval = bool(getattr(self.args, 'disable_train_test_eval', 0))
        test_data, test_loader = (None, None) if disable_train_test_eval else self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, configs=self.args)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            pct_start=self.args.pct_start,
                                            div_factor=10,
                                            anneal_strategy='linear',
                                            epochs=self.args.train_epochs + 1,
                                            steps_per_epoch=self.args.test_every if self.args.test_every else train_steps,
                                            max_lr=self.args.learning_rate)

        epoch_time = time.time()

        best_vali_loss = float('inf')
        best_test_metrics = {'MSE': float('inf'), 'MAE': float('inf'), 'MAPE': float('inf'), 'MSPE': float('inf')}
        min_test_metrics = {'MSE': float('inf'), 'MAE': float('inf'), 'MAPE': float('inf'), 'MSPE': float('inf')}

        for epoch in range(self.args.train_epochs):
            print(f'Starting Training Epoch: {epoch}')
            iter_count = 0
            train_loss = []
            train_specmix_diag = None
            self.model.train()
            for i, data in enumerate(train_loader):
                self.steps += 1
                iter_count += 1
                model_optim.zero_grad()
                batch_x = data[0].float().to(self.device, non_blocking=True)
                batch_y = data[1].float().to(self.device, non_blocking=True)
                batch_x_mark = data[2].float().to(self.device, non_blocking=True)
                batch_y_mark = data[3].float().to(self.device, non_blocking=True)
                pred_len = self.args.pred_len  # batch_y.shape[1] - batch_x.shape[1]
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device,
                                                                                                      non_blocking=True)

                additional_loss = torch.Tensor([0]).to(batch_x.device)

                # Random dropping implementation
                drop_mask = None
                if self.args.random_drop_training:
                    if torch.rand(1).item() > 0:
                        random_drop_rate = torch.rand(1).item()
                        drop_mask = torch.rand(1, 1, batch_x.shape[2], device=batch_x.device) < 1 - random_drop_rate
                        batch_x = batch_x.masked_fill(drop_mask, 0)
                        batch_y = batch_y.masked_fill(drop_mask, 0)
                        batch_x_mark = batch_x_mark.masked_fill(
                            torch.rand(1, 1, batch_x_mark.shape[2], device=batch_x_mark.device) < 1 - random_drop_rate,
                            0)

                self._set_specmix_diagnostics(i == 0)
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if self.args.model == 'STOM':
                            outputs, add_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, train=True)
                            additional_loss += add_loss
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'STOM':
                        outputs, add_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, train=True)
                        additional_loss += add_loss
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # , batch_y

                f_dim = -self.args.number_of_targets
                if self.args.features == 'MS':
                    f_dim = -1
                batch_y = batch_y[:, -pred_len:, f_dim:] if self.args.model != 's2s' else batch_y[:, :, f_dim:]

                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = outputs[:, -pred_len:, f_dim:]
                        MSE = criterion(outputs, batch_y)
                        loss = MSE + additional_loss
                else:
                    outputs = outputs[:, -pred_len:, f_dim:]
                    MSE = criterion(outputs, batch_y)
                    loss = MSE + additional_loss

                wandb.log({
                    'Loss/TrainLossADD': float(additional_loss.item()),
                    'Loss/TrainLossMSE': float(MSE.item()),
                    'Loss/TrainLossTOT': float(loss.item()),
                    'epoch': epoch,
                    'step': self.steps
                })

                train_loss.append(loss.item())
                if (self.args.predictor == 'SpecMix' or getattr(self.args, 'use_stom', 0)) and i == 0:
                    train_specmix_diag = self._get_specmix_diagnostics(reset=True)
                    self._set_specmix_diagnostics(False)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if self.args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    if (iter_count + 1) % self.args.gradient_accumulation == 0:
                        scaler.step(model_optim)
                        scaler.update()
                        model_optim.zero_grad()

                else:
                    loss.backward()
                    if self.args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    if (iter_count + 1) % self.args.gradient_accumulation == 0:
                        model_optim.step()
                        model_optim.zero_grad()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                if self.test_every != 0 and self.steps % self.test_every == 0:
                    print("Test Steps: {} cost time: {}".format(self.test_every, time.time() - epoch_time))
                    wandb.log({'LR/LearningRate': float(scheduler.get_last_lr()[0]), 'step': self.steps})

                    tl = np.average(train_loss)
                    vali_loss, vali_metrics = self.vali(vali_data, vali_loader, criterion)
                    print('Validation Finished (Vali)')
                    if disable_train_test_eval:
                        test_loss = float('nan')
                        test_metrics = {'MSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'),
                                        'MSPE': float('nan'), 'SpecMixDiag': None}
                        print('Validation Finished (Test skipped by disable_train_test_eval)')
                    else:
                        test_loss, test_metrics = self.vali(test_data, test_loader, criterion, label='test')
                        print('Validation Finished (Test)')

                        for metric_name in min_test_metrics:
                            if test_metrics[metric_name] < min_test_metrics[metric_name]:
                                min_test_metrics[metric_name] = test_metrics[metric_name]

                    if vali_loss < best_vali_loss:
                        best_vali_loss = vali_loss
                        best_test_metrics = test_metrics.copy()

                    print(model_optim)

                    print(
                        "Test Steps: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                            self.steps, train_steps, tl, vali_loss, test_loss))
                    specmix_diag = test_metrics.get('SpecMixDiag') or train_specmix_diag
                    self._write_specmix_epoch_log(epoch, tl, vali_loss, test_loss, scheduler.get_last_lr()[0],
                                                  specmix_diag)
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        self.early_stop = True

            # Test on every epochs
            if self.test_every == 0:
                print("Test Steps: {} cost time: {}".format(self.test_every, time.time() - epoch_time))
                wandb.log({'LR/LearningRate': float(scheduler.get_last_lr()[0]), 'step': self.steps})

                tl = np.average(train_loss)
                vali_loss, vali_metrics = self.vali(vali_data, vali_loader, criterion)
                print('Validation Finished (Vali)')
                if disable_train_test_eval:
                    test_loss = float('nan')
                    test_metrics = {'MSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'),
                                    'MSPE': float('nan'), 'SpecMixDiag': None}
                    print('Validation Finished (Test skipped by disable_train_test_eval)')
                else:
                    test_loss, test_metrics = self.vali(test_data, test_loader, criterion, label='test')
                    print('Validation Finished (Test)')

                    for metric_name in min_test_metrics:
                        if test_metrics[metric_name] < min_test_metrics[metric_name]:
                            min_test_metrics[metric_name] = test_metrics[metric_name]

                if vali_loss < best_vali_loss:
                    best_vali_loss = vali_loss
                    best_test_metrics = test_metrics.copy()

                print(model_optim)

                print("Test Steps: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    self.steps, train_steps, tl, vali_loss, test_loss))
                specmix_diag = test_metrics.get('SpecMixDiag') or train_specmix_diag
                self._write_specmix_epoch_log(epoch, tl, vali_loss, test_loss, scheduler.get_last_lr()[0], specmix_diag)
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    self.early_stop = True

            model_optim.zero_grad()
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            if self.early_stop:
                break

            # Refresh train_loader
            train_data, train_loader = self._get_data(flag='train')

        best_model_path = path + '/' + 'checkpoint.pth'
        state_dict = torch.load(best_model_path)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict, strict=False)

        table = wandb.Table(columns=["Metric Type", "MSE", "MAE", "MAPE", "MSPE"])
        table.add_data(
            "Best Vali Model (Test Set)",
            best_test_metrics['MSE'],
            best_test_metrics['MAE'],
            best_test_metrics['MAPE'],
            best_test_metrics['MSPE']
        )
        table.add_data(
            "Min Test Curve Values",
            min_test_metrics['MSE'],
            min_test_metrics['MAE'],
            min_test_metrics['MAPE'],
            min_test_metrics['MSPE']
        )
        wandb.log({"Test Performance Summary": table})

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            state_dict = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            self.model.load_state_dict(new_state_dict, strict=False)

        save_predictions = bool(getattr(self.args, 'save_test_predictions', 1))
        save_visuals = bool(getattr(self.args, 'save_test_visuals', 1))
        preds = []
        trues = []
        inputx = []
        stream_abs = 0.0
        stream_sq = 0.0
        stream_ape = 0.0
        stream_spe = 0.0
        stream_count = 0
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                batch_x = data[0].float().to(self.device)
                batch_y = data[1].float().to(self.device)
                batch_x_mark = data[2].float().to(self.device)
                batch_y_mark = data[3].float().to(self.device)

                pred_len = self.args.pred_len  # batch_y.shape[1] - batch_x.shape[1]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                self._set_specmix_diagnostics(i == 0)
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -self.args.number_of_targets
                if self.args.features == 'MS':
                    f_dim = -1
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)
                outputs = outputs.float().detach().cpu().numpy()
                batch_y = batch_y.float().detach().cpu().numpy()

                pred = outputs
                true = batch_y

                if save_predictions:
                    preds.append(pred)
                    trues.append(true)
                    inputx.append(batch_x.float().detach().cpu().numpy())
                else:
                    diff = pred - true
                    stream_abs += float(np.abs(diff).sum())
                    stream_sq += float(np.square(diff).sum())
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = diff / true
                    stream_ape += float(np.abs(ratio).sum())
                    stream_spe += float(np.square(ratio).sum())
                    stream_count += int(diff.size)
                if save_visuals and i % 20 == 0:
                    input = batch_x.float().detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        if save_predictions:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            inputx = np.concatenate(inputx, axis=0)
            print(preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        else:
            denom = max(stream_count, 1)
            mae = stream_abs / denom
            mse = stream_sq / denom
            rmse = np.sqrt(mse)
            mape = stream_ape / denom
            mspe = stream_spe / denom
            rse = float('nan')
            corr = float('nan')
            print('stream_metrics', {'MSE': mse, 'MAE': mae, 'count': stream_count})

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.close()

        def _metric_scalar(value):
            arr = np.asarray(value)
            if arr.size == 0:
                return float('nan')
            return float(np.nanmean(arr))

        metrics_payload = {
            'MSE': _metric_scalar(mse), 'MAE': _metric_scalar(mae), 'RMSE': _metric_scalar(rmse),
            'MAPE': _metric_scalar(mape), 'MSPE': _metric_scalar(mspe), 'RSE': _metric_scalar(rse),
            'CORR': _metric_scalar(corr),
            'save_test_predictions': int(save_predictions), 'stream_count': int(stream_count),
        }
        with open(folder_path + 'metrics.json', 'w') as f:
            json.dump(metrics_payload, f, indent=2)
        if save_predictions:
            np.save(folder_path + 'pred.npy', preds)
        final_diag = self._get_specmix_diagnostics(reset=True)
        self._set_specmix_diagnostics(False)
        self._write_specmix_final_diagnostics(setting, final_diag)

        # wandb.finish() 
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(pred_loader):
                batch_x = data[0].float().to(self.device)
                batch_y = data[1].float()
                batch_x_mark = data[2].float().to(self.device)
                batch_y_mark = data[3].float().to(self.device)
                pred_len = self.args.pred_len  # batch_y.shape[1] - batch_x.shape[1]

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy(force=True)  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return