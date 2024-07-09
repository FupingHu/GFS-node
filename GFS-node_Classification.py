import time
import warnings
import os
import numpy as np
import pandas
import torch as torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from Args_NodePrediction import get_citation_args
from Utils.Metrics import accuracy
from Utils.Model import GFS_node_classification
from Utils.Util import get_antecedent_parameters, Mu_Norm_List
from Utils.Util import load_citation_network, load_knowledge_graph, GFS_node_Preprocess

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = get_citation_args()
    Time = int(round(time.time() * 1000))
    logger.info('************ Start ************')
    Datasets = ['cora', 'citeseer', 'pubmed', 'nell']
    NumberOfRuns = 10
    Rules = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    L2s = [5e-6]
    Lrs = [0.2]
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
                                        if args.dataset == 'nell':
                                            adj, features, labels, idx_train, idx_val, idx_test = load_knowledge_graph(
                                                args.dataset, 0.001, args.normalization,
                                                args.cuda)
                                        else:
                                            adj, features, labels, idx_train, idx_val, idx_test = load_citation_network(
                                                args.dataset, args.normalization,
                                                args.cuda)
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
                                        features, precompute_time = GFS_node_Preprocess(features, adj, Degree, alpha, beta,
                                                                                    gamma)
                                        VCNs = get_antecedent_parameters(features[idx_train], args.rules)
                                        VCNs = torch.FloatTensor(VCNs).float()
                                        mu_norm_list_train, mu_norm_list_val, mu_norm_list_test = Mu_Norm_List(features,
                                                                                                               VCNs,
                                                                                                               idx_train,
                                                                                                               idx_val,
                                                                                                               idx_test)
                                        train_features = features[idx_train]
                                        train_labels = labels[idx_train]
                                        val_features = features[idx_val]
                                        val_labels = labels[idx_val]
                                        test_features = features[idx_test]
                                        test_labels = labels[idx_test]
                                        logger.info(args)
                                        n_class = labels.max().item() + 1
                                        model = GFS_node_classification(train_features.size(1), n_class, args.rules, args.depth_layers,
                                                      args.hidden)
                                        model.cuda()
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
                                            loss_train = F.cross_entropy(y_train, train_labels)
                                            loss_train.backward()
                                            optimizer.step()
                                            acc_train = accuracy(y_train, train_labels)

                                            model.eval()
                                            res_val = model(val_features)
                                            y_val = torch.cat([res_val_value.unsqueeze(2) for res_val_value in res_val],
                                                              2)
                                            y_val = torch.matmul(y_val, mu_norm_list_val)
                                            y_val = y_val.squeeze(2)
                                            loss_val = F.cross_entropy(y_val, val_labels)
                                            acc_val = accuracy(y_val, val_labels)

                                            res_test = model(test_features)
                                            y_test = torch.cat(
                                                [res_test_value.unsqueeze(2) for res_test_value in res_test], 2)
                                            y_test = torch.matmul(y_test, mu_norm_list_test)
                                            y_test = y_test.squeeze(2)
                                            loss_test = F.cross_entropy(y_test, test_labels)
                                            acc_test = accuracy(y_test, test_labels)

                                            record_dict[epoch] = {'loss_train': loss_train.item(),
                                                                  'acc_train': acc_train.item(),
                                                                  'loss_val': loss_val.item(),
                                                                  'acc_val': acc_val.item(),
                                                                  'loss_test': loss_test.item(),
                                                                  'acc_test': acc_test.item(),
                                                                  'time': time.time() - t}

                                            logger.info('Epoch: {:04d}/{:04d}'.format(epoch + 1, args.epochs) +
                                                        ' loss_train: {:.4f}'.format(loss_train.item()) +
                                                        ' acc_train: {:.4f}'.format(acc_train.item()) +
                                                        ' loss_val: {:.4f}'.format(loss_val.item()) +
                                                        ' acc_val: {:.4f}'.format(acc_val.item()) +
                                                        ' loss_test: {:.4f}'.format(loss_test.item()) +
                                                        ' acc_test: {:.4f}'.format(acc_test.item()) +
                                                        ' time: {:.4f}s'.format(time.time() - t))

                                        logger.info("Optimization Finished!")
                                        logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
                                        logger.info("Best testing performance {: 4f}".format(
                                            np.max([record_dict[epoch]['acc_test'] for epoch in record_dict])))
                                        # logger.info("Minimun loss".format(
                                        #     np.max([record_dict[epoch]['loss_test'] for epoch in record_dict])))
                                        best_test_acc = np.max(
                                            [record_dict[epoch]['acc_test'] for epoch in record_dict])
                                        if not os.path.exists("./results/txt/{}".format(args.dataset)):
                                            os.mkdir("./results/txt/{}".format(args.dataset))
                                        record_txt = open(
                                            "./results/txt/{}/Record_{}_{}_Rule-{}_{}.txt".format(args.dataset, num_run,
                                                                                                  rule, args.dataset,
                                                                                                  TimeStr), "w")
                                        record_txt.write(
                                            'Epoch loss_train acc_train loss_val acc_val loss_test acc_test time ' + '\n')
                                        for record in record_dict:
                                            temp_dict = record_dict[record]
                                            line_txt = str(record) + ' '
                                            for k, v in temp_dict.items():
                                                line_txt += str(v) + ' '
                                            line_txt += '\n'
                                            record_txt.write(line_txt)
                                        record_txt.close()
                                        if not os.path.exists(
                                                './results/csv/{}/{}_{}.csv'.format(args.dataset, args.dataset,
                                                                                    TimeStr)):
                                            pandas.DataFrame(
                                                [[args.rules, args.depth_layers, args.weight_decay, args.lr,
                                                  args.hidden, Degree,
                                                  alpha, beta,
                                                  best_test_acc, params, time.time() - t_total]],
                                                columns=['Number of fuzzy rules', 'Depth of GNN layers',
                                                         'Weight decay(L2)',
                                                         'Learning rate', 'Hidden dims', 'Degree', 'Alpha', 'Beta',
                                                         'Best testing acc',
                                                         'Number of params', 'Total training time']).to_csv(
                                                './results/csv/{}/{}_{}.csv'.format(
                                                    args.dataset, args.dataset, TimeStr), index=False, mode='a')
                                        else:
                                            pandas.DataFrame(
                                                [[args.rules, args.depth_layers, args.weight_decay, args.lr,
                                                  args.hidden, Degree,
                                                  alpha, beta,
                                                  best_test_acc, params, time.time() - t_total]],
                                                columns=['Number of fuzzy rules', 'Depth of GNN layers',
                                                         'Weight decay(L2)',
                                                         'Learning rate', 'Hidden dims', 'Degree', 'Alpha', 'Beta',
                                                         'Best testing acc',
                                                         'Number of params', 'Total training time']).to_csv(
                                                './results/csv/{}/{}_{}.csv'.format(
                                                    args.dataset, args.dataset, TimeStr), index=False, mode='a',
                                                header=False)

    logger.info('************ End ************')
