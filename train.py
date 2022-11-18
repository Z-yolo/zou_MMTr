import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from model import DialogueGCNModel
import pandas as pd
import pickle as pk
import datetime
import ipdb
import seaborn as sns
import matplotlib.pyplot as plt

# from data import IEMOCAPDataset, MMGCNDataset, MMGCNDataloader
from dataloader import IEMOCAPDataset, MELDDataset
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
import logging
# We use seed = 2022 for reproduction of the results reported in the paper.
seed = 2022


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('MELD_features/MELD_features_raw.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('MELD_features/MELD_features_raw.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader



def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False,
                              dataset='IEMOCAP'):
    losses, preds, labels, masks, logits = [], [], [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything()
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        # textf, visuf, acouf, qmask, umask, label, length_mm = [d.cuda() for d in data] if cuda else data[:-1]
        textf, visuf, acouf, qmask, umask, labels_dag, lengths_dag, text_feature = [d.cuda() for d in data] if cuda else data[:-1]
        if args.multi_modal:
            if args.mm_fusion_mthd == 'concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf], dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf], dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf], dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf], dim=-1)
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd == 'gated':
                textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                textf = textf
            else:
                raise NotImplementedError

        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if args.multi_modal and args.mm_fusion_mthd == 'gated':
            log_prob, e_i, e_n, e_t, e_l = model(text_feature, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd == 'concat_subsequently':
            # log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths, acouf, visuf)
            log_prob = model(text_feature, qmask, umask, lengths, acouf, visuf,  lengths_dag)
        else:
            log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)


        loss = loss_function(log_prob.permute(1, 2, 0), labels_dag)
        label = labels_dag.cpu().numpy().tolist()
        pred = torch.argmax(log_prob.transpose(1, 0), dim=2).cpu().numpy().tolist()  # .transpose(1, 0)
        preds += pred
        labels += label
        losses.append(loss.item())
        logits.append(log_prob)


        max_grad_norm = 2.0
        if train:
            loss_val = loss.item()
            loss.backward()
            # flood.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


    # if dataset == "IEMOCAP":

    if preds != []:
        new_preds = []
        new_labels = []
        for i, label in enumerate(labels):
            for j, l in enumerate(label):
                if l != -1:
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    # avg_predlog = round(np.sum(logits) / len(logits))

    return avg_loss, avg_accuracy, new_labels, new_preds, avg_fscore


if __name__ == '__main__':
    path = './models_logging/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=False,
                        help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')

    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')

    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')

    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='relation', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=5, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False,
                        help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=True,
                        help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=False,
                        help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='concat',
                        help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False,
                        help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=256, help='Deep_GCN_nlayers')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=False, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--patience', type=int, default=5, help='early stop')
    # new add

    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='number of chunks per batch (default: 1)')
    parser.add_argument('--nlevels', type=int, default=5,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--num_heads', type=int, default=5,
                        help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')

    # Dropouts
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                        help='attention dropout (for audio)')
    parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                        help='attention dropout (for visual)')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0,
                        help='output layer dropout')
    parser.add_argument('--dropout_crn', type=float, default=0.2, metavar='dropout', help='dropout rate')



    args = parser.parse_args()
    hyp_params = args
    today = datetime.datetime.now()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()
    logger = get_logger(path + args.Dataset + "/{}_{}_{}_logging.log".format(today.date(), today.hour, today.minute))
    logging.info(args)
    if args.cuda:
        logger.info('Running on GPU')
    else:
        logger.info('Running on CPU')
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600,
                'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = feat2dim['textCNN'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.mm_fusion_mthd == 'concat':
            if modals == 'avl':
                D_m = D_audio + D_visual + D_text
            elif modals == 'av':
                D_m = D_audio + D_visual
            elif modals == 'al':
                D_m = D_audio + D_text
            elif modals == 'vl':
                D_m = D_visual + D_text
            else:
                raise NotImplementedError
        else:
            D_m = D_text
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1
    hidden_size, input_size = 100, 100
    logger.info([args.lr, args.batch_size, args.l2])
    if args.graph_model:
        seed_everything()

        model = DialogueGCNModel(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v=D_visual,
                                 D_m_a=D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal
                                 , hyp_params=hyp_params
                                 )

        # print('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    else:
        name = 'Base'

    if cuda:
        model.cuda()
    logging.info("The model have {} parameters in total".format(sum(x.numel() for x in model.parameters())))
    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1 / 0.086747,
                                          1 / 0.144406,
                                          1 / 0.227883,
                                          1 / 0.160585,
                                          1 / 0.127711,
                                          1 / 0.252668])

    if args.Dataset == 'MELD':
        class_weights = torch.FloatTensor(
            [1 / 0.469506857, 1 / 0.119346367, 1 / 0.026116137,
             1 / 0.073096002, 1 / 0.168368836, 1 / 0.026334987,
             1 / 0.117230814])
        class_weights = class_weights.cuda()
        # loss_function = FocalLoss(gamma=args.gamma, alpha=class_weights if args.class_weight else None)
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        if args.class_weight:
            if args.graph_model:
                # loss_function1 = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
                loss_function = nn.CrossEntropyLoss(ignore_index=-1)
            else:
                loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                # loss_function1 = nn.NLLLoss()
                loss_function = nn.CrossEntropyLoss(ignore_index=-1)
            else:
                loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    if args.Dataset =="MELD":
        save_model_path = "saved_model/{}/pytorch_model.bin".format(args.Dataset)
    else:
        save_model_path = "saved_model/{}/pytorch_model.bin".format(args.Dataset)

    torch.save(model.state_dict(), save_model_path)

    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                   batch_size=batch_size,
                                                                   num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask, patience, best_epoch = None, None, None, None, None, 0, -1
    all_fscore, all_acc, all_loss = [], [], []
    all_labels, all_preds = [], []
    test_label = False
    if test_label:
        state = torch.load('saved_model/pytorch_model.bin')
        model.load_state_dict(state['net'])
        test_loss, test_acc, test_label, test_pred, test_fscore, avg_macro_fscore = train_or_eval_graph_model(model,
                                                                                                              loss_function,
                                                                                                              test_loader,
                                                                                                              0, cuda,
                                                                                                              args.modals,
                                                                                                              dataset=args.Dataset)

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore = train_or_eval_graph_model(model, loss_function, train_loader,
                                                                                  e, cuda, args.modals, optimizer, True,
                                                                                  dataset=args.Dataset)
            # valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_graph_model(model, loss_function2, valid_loader, e, cuda, args.modals, dataset=args.Dataset)
            test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_graph_model(model, loss_function,
                                                                                                test_loader, e, cuda,
                                                                                                args.modals,
                                                                                                dataset=args.Dataset)
            all_fscore.append(test_fscore)
            all_acc.append(test_acc)
            all_preds.append(test_pred)
            all_labels.append(test_label)
        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, patience = test_loss, test_label, test_pred, 0
        else:
            patience +=1

        if best_fscore == None or e == 0 or best_fscore < test_fscore:
            best_epoch, best_fscore = e, test_fscore
            test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_graph_model(model, loss_function,
                                                                                                test_loader, e, cuda,
                                                                                                args.modals,
                                                                                                dataset=args.Dataset)

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        logger.info(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
            format(e + 1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore,
                   round(time.time() - start_time, 2)))

        if (e+1) % 10 == 0:
            logger.info(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            logger.info(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + 'using_lstm_' + args.Dataset
    else:
        name_ = args.mm_fusion_mthd + '_' + args.modals + '_' + args.graph_type + '_' + args.graph_construct + str(
            args.Deep_GCN_nlayers) + '_' + args.Dataset
    if patience >= args.patience:
        logger.info("Early stopping....{}".format(patience))

    if args.use_speaker:
        name_ = name_ + '_speaker'
    if args.use_modal:
        name_ = name_ + '_modal'
    if args.tensorboard:
        writer.close()
    a = np.argmax(all_fscore)
    logger.info('Test performance..')
    logger.info('F-Score:{}, Accuracy:{}, best_epoch:{}'.format(max(all_fscore), all_acc[a], best_epoch + 1))
    # print("Expected Labels:", all_labels[a])
    # print("Best F1 Score prediction:", all_preds[a])
    print("F1 score:", all_fscore)
    # print("data logit:", all_logit[a])

    if not os.path.exists("results_record/record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
        with open("results_record/record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
            pk.dump({}, f)
    with open("results_record/record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
        record = pk.load(f)
    key_ = name_
    if record.get(key_, False):
        record[key_].append({"F-score": max(all_fscore)})
        record[key_].append({"Accuracy": max(all_acc)})
    else:
        record[key_] = [max(all_fscore)]
    if record.get(key_+'record', False):
        record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    else:
        record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)]
    with open("results_record/record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'wb') as f:
        pk.dump(record, f)

    C = confusion_matrix(best_label, best_pred, sample_weight=best_mask)
    # np.set_printoptions(precision=2)
    cm_normalized = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.round(cm_normalized, 2)
    if args.Dataset == 'IEMOCAP':
        df = pd.DataFrame(cm_normalized, index=["Hap", "Sad", "Neu", "Ang", "Exc", "Fru"],
                          columns=["Hap", "Sad", "Neu", "Ang", "Exc", "Fru"])
    else:
        df = pd.DataFrame(cm_normalized, index=["Neu", "Sur", "Fea", "Sad", "Joy", "Dis", "Ang"],
                          columns=["Neu", "Sur", "Fea", "Sad", "Joy", "Dis", "Ang"])
    # plt.figure(figsize=(6, 6))
    # sns.set_theme(style="whitegrid", font='Times New Roman', font_scale=1.4)
    # sns.heatmap(df, annot=True, cmap="Blues", square=True, cbar=False) # , fmt='.20g'
    # # plt.savefig(r"C:\Users\1234\Desktop\邹\论文_邹\iemocap_heatmap_ours.pdf", bbox_inches="tight")
    # plt.show()
    logger.info(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    logger.info(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
    logger.info(cm_normalized)


