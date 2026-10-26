import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='STOM: Source-Traceable Operator Mixtures for Long Horizon Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2026, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='STOM',
                    help='model name')
parser.add_argument('--model_type', type=str, default='DL', help='model type [DL]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--scale', type=int, default=1, help='standardization')
parser.add_argument('--train_ratio', type=float, default=0.7, help='training set')
parser.add_argument('--test_ratio', type=float, default=0.2, help='test set')
parser.add_argument('--plot_every', type=int, default=10, help='visualization')
parser.add_argument('--iterative_prediction', type=int, default=0, help='recursively predict')
parser.add_argument('--number_of_targets', type=int, default=0, help='partial output')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=0, help='dimension of model; 0 uses the automatic STOM backbone dimension')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.00, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', default=False, action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--time_emb_dim', type=int, default=4, help='dimension of time embedding')
parser.add_argument('--mlp_ratio', type=int, default=8, help='MLP ratio')

# optimization & traning
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=12, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0008, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.0, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--max_grad_norm', type=float, default=1, help='clip norm')
parser.add_argument('--test_every', type=int, default=0, help='steps between two tests')
parser.add_argument('--gradient_accumulation', type=int, default=1, help='gradient accumulation')
parser.add_argument('--batch_size_test', type=int, default=0, help='batch_size_test_ratio')
parser.add_argument('--predictor', type=str, default='LinAttn', help='LinAttn+RoPE temporal backbone used by STOM')
parser.add_argument('--resume', type=str, default='none', help='resume training from a checkpoint')
parser.add_argument('--compile', type=int, default=0, help='whether to use torch compile')
parser.add_argument('--save_test_predictions', type=int, default=1, help='save final test pred.npy; set 0 for large datasets to save metrics.json only')
parser.add_argument('--save_test_visuals', type=int, default=1, help='save final test visualization PDFs')
parser.add_argument('--disable_train_test_eval', type=int, default=0, help='if 1, do not evaluate test set during training epochs')
parser.add_argument('--skip_final_test_after_train', type=int, default=0, help='if 1, do not run exp.test immediately after training')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
parser.add_argument('--random_drop_training', type=int, default=0, help='random dropping for multivariate models')
parser.add_argument('--model_label', type=str, default='Model', help='model label')

# Autoregressive
parser.add_argument('--univariate', type=int, default=0, help='whether to fall back to univariate AR tokeinzation rather than ARX tokenization')

# TimeMixer
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='dft_decomp',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--channel_mixing', type=int, default=1,
                    help='0: channel mixing 1: whether to use channel_mixing')
parser.add_argument('--percent', type=int, default=100, help='few shot percent')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')

# Random Channel Dropout
parser.add_argument('--random_channel_dropout', type=int, default=1,
                    help='0: Without random_channel_dropout 1: With random_channel_dropout')
parser.add_argument('--weight_decay', type=float, default=0.0, help='AdamW weight decay')

# SpecMix/STOM diagnostics
parser.add_argument('--specmix_training_curve_path', type=str, default='', help='CSV path for SpecMix/STOM training diagnostics')
parser.add_argument('--specmix_tensorboard_dir', type=str, default='', help='TensorBoard directory for SpecMix/STOM diagnostics')
parser.add_argument('--specmix_diagnostics_path', type=str, default='', help='JSON path for final SpecMix/STOM diagnostics')
parser.add_argument('--specmix_gate_reg', type=float, default=0.0, help='reserved; default off')
parser.add_argument('--specmix_gate_rho', type=float, default=0.2, help='reserved gate target')
parser.add_argument('--specmix_gate_reg_weight', type=float, default=1e-4, help='reserved; default unused')
parser.add_argument('--use_stom', type=int, default=1, help='enable the STOM forecast head on the LinAttn+RoPE backbone')
parser.add_argument('--stom_ma_kernel', type=int, default=25, help='STOM moving average kernel size')
parser.add_argument('--stom_local_kernel', type=int, default=5, help='STOM depthwise local kernel size')
parser.add_argument('--stom_spec_gain_init', type=float, default=6.0, help='STOM spectral gain softplus initial value')
parser.add_argument('--stom_gate_prior', type=str, default='0.35,0.10,0.30,0.25', help='STOM gate prior as comma-separated phase,spectral,smooth,local')
parser.add_argument('--stom_gate_hidden_dim', type=int, default=32, help='STOM gate MLP hidden dimension')
parser.add_argument('--stom_rank', type=int, default=4, help='STOM spectral low-rank potential rank')
parser.add_argument('--stom_n_spec_heads', type=int, default=4, help='STOM spectral transfer head count')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = {
        'DL': Exp_Main
    }[args.model_type]

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if not getattr(args, 'skip_final_test_after_train', 0):
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
        else:
            print('>>>>>>>skip final test after training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()