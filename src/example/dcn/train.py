# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2021-01-01 19:08
# Description:  
#--------------------------------------------
import os
import yaml
import time
import argparse
import torch
import random
import numpy as np
from torch import optim
from concurrent.futures import ProcessPoolExecutor
from src.utils import WorkerInitObj
from src.model.dcn import DCN
from src.evaluate import evaluate
from src.dataset import subsetDataloader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization.")
    parser.add_argument('--epoch', default=32, type=int, help='The epoch of train')
    parser.add_argument('--input_dir', default=None, type=str, required=True, help="The train input data dir.")
    parser.add_argument('--output_dir', default=None, type=str, required=True, help="The output of checkpoint dir.")
    parser.add_argument("--init_checkpoint", default=None, type=str, help="The initial checkpoint to start training from.")
    parser.add_argument('--batch_size', default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--deep_layers', nargs='+', type=int, default=[256, 32, 8, 1],  help='The layer hiddens for mlp.')
    parser.add_argument('--embedding_dim', default=8, type=int, help='The dimension of user and item embeddings')
    parser.add_argument('--use_cuda', default=False, action='store_true', help='Whether use gpu')
    parser.add_argument('--devices', type=str, default='0,1', help='The devices id of gpu')
    parser.add_argument('--learning_rate', default=1.5e-3, type=float, help="The initial learning rate for optimizer")
    parser.add_argument('--eval_freq', default=100, type=int, help='The freq of eval test set')
    parser.add_argument('--log_freq', default=30, type=int, help='The freq of print log')

    args = parser.parse_args()

    return args

def setup_training(args):

    args.multi_gpu = False
    if args.use_cuda:
        assert torch.cuda.is_available()

        args.device_ids = [int(id) for id in args.devices.strip().split(',')]
        if len(args.device_ids) > 1:
            assert torch.cuda.device_count() >= len(args.device_ids)
            os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
            device = torch.device('cuda:0')
            args.multi_gpu = True
        else:
            device = torch.device('cuda:{}'.format(args.devices))
            os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    else:
        device = torch.device('cpu')

    train_dir = os.path.join(args.input_dir, 'train')
    val_dir = os.path.join(args.input_dir, 'val')
    args.model_config_path = os.path.join(args.input_dir, 'config.yaml')
    args.train_path = os.path.join(train_dir, [f for f in os.listdir(train_dir) if f.startswith('part')][0])
    args.val_path = os.path.join(val_dir, [f for f in os.listdir(val_dir) if f.startswith('part')][0])

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    return device, args

def prepare_model(args):
    config = yaml.load(open(args.model_config_path), Loader=yaml.FullLoader)
    sparse_feat_and_nums = list(config['sparse'].items())
    dense_feat = list(config['dense'].keys())
    model = DCN(sparse_feat_and_nums=sparse_feat_and_nums,
                cross_layers=args.deep_layers,
                dense_feat=dense_feat,
                embed_dim=args.embedding_dim,
                deep_layers=args.deep_layers,
                sigmoid_out=True)

    return model

def prepare_model_and_optimizer(args, device):
    lr = args.learning_rate

    #model
    model = prepare_model(args)

    args.model_name = model.__class__.__name__
    if args.init_checkpoint is not None and os.path.isfile(args.init_checkpoint):
        checkpoint_name = args.init_checkpoint.split('/')[-1].split('_')[0]
        if checkpoint_name == args.model_name:
            checkpoint = torch.load(args.init_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            print('init weight from {}'.format(args.init_checkpoint))
        else:
            raise ValueError('expect {} model, but get {} model'.format(args.model_name, checkpoint_name))

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = torch.nn.BCELoss()
    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    model.to(device)
    criterion.to(device)

    return model, optimizer, criterion

def main():
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    worker_init = WorkerInitObj(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    device, args = setup_training(args)
    model, optimizer, criterion = prepare_model_and_optimizer(args, device)
    pool = ProcessPoolExecutor(1)
    train_iter = subsetDataloader(path=args.train_path,
                                  batch_size=args.batch_size,
                                  worker_init=worker_init)
    test_iter = subsetDataloader(path=args.val_path,
                                 batch_size=args.batch_size,
                                 worker_init=worker_init)

    print('-' * 50 + 'args' + '-' * 50)
    for k in list(vars(args).keys()):
        print('{0}: {1}'.format(k, vars(args)[k]))
    print('-' * 30)
    print(model)
    print('-' * 50 + 'args' + '-' * 50)

    global_step = 0
    global_auc = 0

    s_time_train = time.time()
    for epoch in range(args.epoch):

        dataset_future = pool.submit(subsetDataloader,
                                     args.train_path,
                                     args.batch_size,
                                     worker_init)

        for step, batch in enumerate(train_iter):

            model.train()
            labels = batch['label'].to(device).float()
            batch = {t: {k: v.to(device) for k, v in d.items()} for t, d in batch.items() if isinstance(d, dict)}

            optimizer.zero_grad()
            logits = model(batch)
            # print('logits', logits)
            # print('label', labels)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            # evaluate
            if global_step != 0 and global_step % args.eval_freq == 0:
                s_time_eval = time.time()
                model.eval()
                auc = evaluate(model, test_iter, device)
                e_time_eval = time.time()
                print('-' * 68)
                print('Epoch:[{0}] Step:[{1}] AUC:[{2}] time:[{3}s]'.format(
                    epoch,
                    global_step,
                    format(auc, '.4f'),
                    format(e_time_eval - s_time_eval, '.4f')))

                if auc > global_auc:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_save_file = os.path.join(args.output_dir, "{}_auc_{}_step_{}_ckpt.pt".format(
                        args.model_name,
                        format(auc, '.4f'),
                        global_step))

                    if os.path.exists(output_save_file):
                        os.system('rm -rf {}'.format(output_save_file))
                    torch.save({'model': model_to_save.state_dict(),
                                'name': args.model_name},
                               output_save_file)
                    print('Epoch:[{0}] Step:[{1}] SavePath:[{2}]'.format(
                        epoch,
                        global_step,
                        output_save_file))
                    global_auc = auc
                print('-' * 68)

            # log
            if global_step != 0 and global_step % args.log_freq == 0:
                e_time_train = time.time()
                print('Epoch:[{0}] Step:[{1}] Loss:[{2}] Lr:[{3}] time:[{4}s]'.format(
                    epoch,
                    global_step,
                    format(loss.item(), '.4f'),
                    format(optimizer.param_groups[0]['lr'], '.6'),
                    format(e_time_train - s_time_train, '.4f')))
                s_time_train = time.time()

            global_step += 1

        del train_iter
        train_iter = dataset_future.result(timeout=None)


if __name__ == '__main__':
    main()
