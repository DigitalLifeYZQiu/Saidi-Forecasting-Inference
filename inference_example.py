import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.print_args import print_args
import random
import numpy as np

argsDict={
    # 需要配置的参数
    'root_path': './dataset',
    'data_path': 'lab_hourly_average.csv',
    'inference_ckpt_path': 'checkpoints/checkpoint-example.pth',
    # 其他可以默认的参数
    'task_name': 'long_term_forecast',
    'is_training': 1,
    'model_id': 'TSMixer_saidi_24_1',
    'model': 'TSMixer',
    'data': 'saidi',
    'features': 'M',
    'target': 'OT',
    'freq': 'h',
    'checkpoints': './checkpoints/',
    'seq_len': 24,
    'label_len': 1,
    'pred_len': 1,
    'seasonal_patterns': 'Monthly',
    'inverse': False,
    'mask_rate': 0.25,
    'anomaly_ratio': 0.25,
    'expand': 2,
    'd_conv': 4,
    'top_k': 5,
    'num_kernels': 6,
    'enc_in': 6,
    'dec_in': 6,
    'c_out': 6,
    'd_model': 512,
    'n_heads': 8,
    'e_layers': 2,
    'd_layers': 1,
    'd_ff': 2048,
    'moving_avg': 25,
    'factor': 3,
    'distil': True,
    'dropout': 0.1,
    'embed': 'timeF',
    'activation': 'gelu',
    'output_attention': False,
    'channel_independence': 1,
    'decomp_method': 'moving_avg',
    'use_norm': 1,
    'down_sampling_layers': 0,
    'down_sampling_window': 1,
    'down_sampling_method': None,
    'seg_len': 48,
    'num_workers': 10,
    'itr': 1,
    'train_epochs': 10,
    'batch_size': 32,
    'patience': 5,
    'learning_rate': 0.0001,
    'des': 'Exp',
    'loss': 'MSE',
    'lradj': 'type1',
    'use_amp': False,
    # 'use_gpu': True,
    'use_gpu': False,
    'use_multi_gpu': False,
    'p_hidden_dims': [128, 128],
    'p_hidden_layers': 2,
    'seed': 2,
}
args = argparse.Namespace(**argsDict)
print(args)
if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast
elif args.task_name == 'anomaly_detection':
    Exp = Exp_Anomaly_Detection
else:
    Exp = Exp_Long_Term_Forecast

exp = Exp(args)

# forecasting_model = exp.model.load_state_dict(torch.load('./checkpoints/checkpoint-example.pth'))

def load_model(args, exp):
    # 模型结构在exp初始化时已经创建，这里只对exp类中模型导入预训练参数即可
    model = exp.model.load_state_dict(torch.load(args.inference_ckpt_path))
    return model

model  = load_model(args, exp)

# 将输入的list转换为tensor形式
def list_to_input_tensor(input_list):
    input = np.array(input_list)
    if len(input.shape)<3:
        input = np.expand_dims(input, axis=0)
        input = torch.Tensor(input).to(exp.device)
    else:
        input = torch.Tensor(input).to(exp.device)
    return input

def output_tensor_to_list(output_tensor):
    if output_tensor.device.type == 'cuda':
        output = output_tensor.detach().to('cpu')
    else:
        output = output_tensor
    # * Not using inverse transform on dataset level
    # if inference_data.scale and exp.args.inverse:
    #     shape = output.shape
    #     inference_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
    if len(output.shape)==3:
        output = output.reshape(output.shape[1],output.shape[2])
    return output.tolist()

# 数据集导入(dataset \ dataloader)
inference_data, inference_loader = exp._get_data(flag='test')

# 获取数据
#   输入数据：sample（1，24，6）
#   输出数据：outputs（1，1，6）
#   回归标签：label（1，2，6） - 包含label_len与pred_len

index=1
sample_list = inference_data[index][0].tolist()
label_list = inference_data[index][1].tolist()
sample = list_to_input_tensor(sample_list)
label = list_to_input_tensor(label_list)

outputs = exp.inference(args = args, sample=sample)

predict_outputs = output_tensor_to_list(outputs)



import pdb
pdb.set_trace()