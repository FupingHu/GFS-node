import os
import time
import warnings
import numpy as np
import pandas
import torch as torch
import torch.optim as optim
from loguru import logger
from Args_NodePrediction import get_citation_args
from Utils.Model import GFS_node_regression
from Utils.Util import get_antecedent_parameters, Mu_Norm_List
from Utils.Util import load_data_regression

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = get_citation_args()
    Time = int(round(time.time() * 1000))
    logger.info('************ Start ************')
    Datasets = ['chameleon', 'crocodile', 'squirrel']
    NumberOfRuns = 10
    Rules = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Rules = [6]
    L2s = [5e-6]
    Lrs = [2e-1]
    Hiddens = [32, 64, 128, 256, 512]
    Depths = [2]
    Alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    Degrees = [2]
    record_dict = {}
    for num_run in range(NumberOfRuns):
        TimeStr = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(Time / 1000))
        for dataset in Datasets:
            args.dataset = dataset
            for rule in Rules:
                for l2 in L2s:
                    for lr in Lrs:
                        for hidden in Hiddens:
                            for depth in Depths:
                                for Alpha in Alphas:
                                    for Degree in Degrees:
                                        features, targets, idx_train, idx_val, idx_test = load_data_regression(
                                            args.dataset)
                                        logger.add(
                                            "./results/logs/{}/{}_{}.log".format(args.dataset, args.dataset, TimeStr))
                                        logger.info(
                                            "Train/Dev/Test Nodes: {}/{}/{}".format(len(idx_train), len(idx_val),
                                                                                    len(idx_test)))
                                        alpha = Alpha
                                        beta = 1 - alpha
                                        gamma = 0
                                        args.rules = rule
                                        args.depth_layers = depth
                                        args.hidden = hidden
                                        args.lr = lr
                                        args.weight_decay = l2

                                        VCNs = get_antecedent_parameters(features[idx_train], args.rules)

                                        VCNS = torch.FloatTensor(VCNs).float()

                                        mu_norm_list_train, mu_norm_list_val, mu_norm_list_test = Mu_Norm_List(features,
                                                                                                               VCNS,
                                                                                                               idx_train,
                                                                                                               idx_val,
                                                                                                               idx_test)

                                        train_features = features[idx_train]
                                        train_targets = targets[idx_train]
                                        val_features = features[idx_val]
                                        val_targets = targets[idx_val]
                                        test_features = features[idx_test]
                                        test_targets = targets[idx_test]
                                        model = GFS_node_regression(train_features.size(1), args.rules,
                                                                    args.depth_layers,
                                                                    args.hidden)
                                        params = sum(p.numel() for p in list(model.parameters()))
                                        logger.info('#Number of Params:%f' % params)
                                        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                                               weight_decay=args.weight_decay)
                                        t_total = time.time()
                                        for epoch in range(args.epochs):
                                            t = time.time()
                                            model.train()
                                            optimizer.zero_grad()
                                            res = model(train_features)
                                            y_train = torch.cat([y_pred_rule.unsqueeze(2) for y_pred_rule in res], 2)
                                            y_train = torch.matmul(y_train, mu_norm_list_train)
                                            y_train = y_train.squeeze(2)
                                            y_train = y_train.squeeze(1)
                                            mse = torch.nn.MSELoss()
                                            loss_train = mse(y_train, train_targets)
                                            loss_train.backward()
                                            optimizer.step()
                                            MSE_train = mse(y_train, train_targets)

                                            model.eval()
                                            res_val = model(val_features)
                                            y_val = torch.cat([res_val_value.unsqueeze(2) for res_val_value in res_val],
                                                              2)
                                            y_val = torch.matmul(y_val, mu_norm_list_val)
                                            y_val = y_val.squeeze(2)
                                            y_val = y_val.squeeze(1)
                                            loss_val = mse(y_val, val_targets)
                                            MSE_val = mse(y_val, val_targets)

                                            res_test = model(test_features)
                                            y_test = torch.cat(
                                                [res_test_value.unsqueeze(2) for res_test_value in res_test], 2)
                                            y_test = torch.matmul(y_test, mu_norm_list_test)
                                            y_test = y_test.squeeze(2)
                                            y_test = y_test.squeeze(1)
                                            loss_test = mse(y_test, test_targets)
                                            MSE_test = mse(y_test, test_targets)

                                            record_dict[epoch] = {'loss_train': loss_train.item(),
                                                                  'MSE_train': MSE_train.item(),
                                                                  'loss_val': loss_val.item(),
                                                                  'MSE_val': MSE_val.item(),
                                                                  'loss_test': loss_test.item(),
                                                                  'MSE_test': MSE_test.item(),
                                                                  'time': time.time() - t}

                                            logger.info('Epoch: {:04d}/{:04d}'.format(epoch + 1, args.epochs) +
                                                        ' loss_train: {:.4f}'.format(loss_train.item()) +
                                                        ' MSE_train: {:.4f}'.format(MSE_train.item()) +
                                                        ' loss_val: {:.4f}'.format(loss_val.item()) +
                                                        ' MSE_val: {:.4f}'.format(MSE_val.item()) +
                                                        ' loss_test: {:.4f}'.format(loss_test.item()) +
                                                        ' MSE_test: {:.4f}'.format(MSE_test.item()) +
                                                        ' time: {:.4f}s'.format(time.time() - t))

                                        logger.info("Optimization Finished!")
                                        logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
                                        logger.info("Best testing performance {: 4f}".format(
                                            np.min([record_dict[epoch]['MSE_test'] for epoch in record_dict])))
                                        best_test_MSE = np.max(
                                            [record_dict[epoch]['MSE_test'] for epoch in record_dict])

                                        if not os.path.exists(
                                                './results/csv/{}'.format(args.dataset)):
                                            os.mkdir('./results/csv/{}'.format(args.dataset))
                                            pandas.DataFrame(
                                                [[args.rules, args.depth_layers, args.weight_decay, args.lr,
                                                  args.hidden, Degree,
                                                  alpha, beta,
                                                  best_test_MSE, params, time.time() - t_total]],
                                                columns=['Number of fuzzy rules', 'Depth of GNN layers',
                                                         'Weight decay(L2)',
                                                         'Learning rate', 'Hidden dims', 'Degree', 'Alpha', 'Beta',
                                                         'Best testing MSE',
                                                         'Number of params', 'Total training time']).to_csv(
                                                './results/csv/{}/{}_{}.csv'.format(
                                                    args.dataset, args.dataset, TimeStr), index=False, mode='a')
                                        else:
                                            pandas.DataFrame(
                                                [[args.rules, args.depth_layers, args.weight_decay, args.lr,
                                                  args.hidden, Degree,
                                                  alpha, beta,
                                                  best_test_MSE, params, time.time() - t_total]],
                                                columns=['Number of fuzzy rules', 'Depth of GNN layers',
                                                         'Weight decay(L2)',
                                                         'Learning rate', 'Hidden dims', 'Degree', 'Alpha', 'Beta',
                                                         'Best testing MSE',
                                                         'Number of params', 'Total training time']).to_csv(
                                                './results/csv/{}/{}_{}.csv'.format(
                                                    args.dataset, args.dataset, TimeStr), index=False, mode='a',
                                                header=False)
    logger.info('************ End ************')
