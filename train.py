"""全局map
fire命令行
在一个数据集上进行一次完整的流程：
    加载数据、划分数据集
    训练（evaluate）
    预测
    写结果到json
k_fold流程
* 自动检测空闲GPU保证训练完成
"""
import os
import numpy as np
import json
import logging
import fire
import datetime
from models import *
from data_utils import MyData
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from types import SimpleNamespace

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def run(**kwags):
    with open('conf.json') as f:
        opt = json.load(f)
    for k, w in kwags.items():
        if k not in opt:
            logging.getLogger(__name__).error(f'{k} not in config')
            exit(0)
        else:
            opt[k] = w
    opt['time_stamp'] = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join('out', opt['time_stamp'])
    opt = SimpleNamespace(**opt)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    train, test = [MyData(opt, m) for m in ('train', 'test')]
    train = DataLoader(train,
                       batch_size=opt.train_batch_size,
                       shuffle=True,
                       drop_last=False)
    test = DataLoader(test,
                      batch_size=opt.test_batch_size,
                      shuffle=False,
                      drop_last=False)

    model = TSA_LSTM(opt)
    model = init_weight(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    max_acc = 0
    f1_at_max_acc = 0
    loss_at_max_acc = 0
    loss_at_max_epoch = 0
    tr_acc_at_max_acc = 0
    tr_f1_at_max_acc = 0
    tr_acc_at_max_epoch = 0
    tr_f1_at_max_epoch = 0
    epoch_at_max_acc = 0

    for e in range(1, opt.num_epoch + 1):
        model.train()
        tr_preds, tr_labels, tr_losses = [], [], []
        for i, batch in enumerate(train):
            model.zero_grad()
            x = [batch[c] for c in input_cols[opt.model_name]]
            y = batch['polarity']
            if not opt.debug:
                x = [e.to('cuda') for e in x]
                y = y.to('cuda')
            tr_labels += y.tolist()
            predict = model(x)
            tr_preds += predict.max(-1)[1].tolist()
            loss = loss_fn(predict, y)
            tr_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if i % opt.print_every == 0:
                logging.getLogger(__name__).info(f'loss: {loss.item():.2f}')
        tr_loss, tr_f1, tr_acc = np.mean(tr_losses), f1_score(
            tr_labels, tr_preds,
            average='macro'), accuracy_score(tr_labels, tr_preds)
        with torch.no_grad():
            preds = []
            labels = []
            probabilities = []
            model.eval()
            for batch in test:
                x = [batch[c] for c in input_cols[opt.model_name]]
                if not opt.debug:
                    x = [e.to('cuda') for e in x]
                y = batch['polarity']
                labels += y.tolist()
                probs = model(x)
                probabilities += probs.tolist()
                p = probs.max(-1)[1].tolist()
                preds += p
            acc, f1 = accuracy_score(labels, preds), f1_score(labels,
                                                              preds,
                                                              average='macro')

            if acc > max_acc:
                logging.getLogger(__name__).info(
                    f'Improved epoch: {e} acc: {acc:.2f} f1: {f1:.2f}')
                max_acc = acc
                f1_at_max_acc = f1
                loss_at_max_acc = tr_loss
                tr_acc_at_max_acc = tr_acc
                tr_f1_at_max_acc = tr_f1
                epoch_at_max_acc = e
                with open(os.path.join(out_dir, 'predictions.csv'), 'w') as f:
                    for pro, pred, label in zip(probabilities, preds, labels):
                        f.write(
                            str(label) + ', ' + str(pred) + ', ' +
                            ', '.join([str(p) for p in pro]) + '\n')
            else:
                logging.getLogger(__name__).info(
                    f'Not Improved epoch: {e} acc: {acc:.2f} f1: {f1:.2f}')
        loss_at_max_epoch = tr_loss
        tr_f1_at_max_acc = tr_acc
        tr_acc_at_max_acc = tr_f1

    opt = vars(opt)
    opt['max_acc'] = max_acc
    opt['f1_at_max_acc'] = f1_at_max_acc
    opt['loss_at_max_acc'] = loss_at_max_acc
    opt['loss_at_max_epoch'] = loss_at_max_epoch
    opt['tr_acc_at_max_acc'] = tr_acc_at_max_acc
    opt['tr_f1_at_max_acc'] = tr_f1_at_max_acc
    opt['tr_acc_at_max_epoch'] = tr_acc_at_max_epoch
    opt['tr_f1_at_max_epoch'] = tr_f1_at_max_epoch
    opt['epoch_at_max_acc'] = epoch_at_max_acc

    with open(os.path.join(out_dir, 'conf.json'), 'w') as f:
        json.dump(opt, f, indent=4, ensure_ascii=False)
    logging.getLogger(__name__).info('done !')


if __name__ == "__main__":
    # run(debug=True, num_epoch=10, train_batch_size=2, test_batch_size=2)
    fire.Fire()