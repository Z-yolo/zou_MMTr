import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math
from model_GCN import GCN_2Layers, GCNLayer1, GCNII, TextCNN
from model_mm import MM_GCN, MM_GCN2
from modules.transformer import TransformerEncoder
import ipdb

class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1,2,0)
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim) cand_dim == mem_dim?
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            M_ = M.permute(1,2,0)
            x_ = x.unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0)
            x_ = self.transform(x).unsqueeze(1)
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            alpha_masked = alpha_*mask.unsqueeze(1)
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)
            alpha = alpha_masked/alpha_sum
        else:
            M_ = M.transpose(0,1)
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1)
            M_x_ = torch.cat([M_,x_],2)
            mx_a = F.tanh(self.transform(M_x_))
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2)

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:]
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


class DialogueRNNCell(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell = nn.GRUCell(D_p,D_e)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p,D_p)

        self.dropout = nn.Dropout(dropout)

        if context_attention=='simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U,q0_sel], dim=1),
                torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0 else
                g_hist[-1])
        g_ = self.dropout(g_)
        if g_hist.size()[0]==0:
            c_ = torch.zeros(U.size()[0],self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist,U)
        U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = self.p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
            U_ss_ = torch.cat([U_,ss_],1)
            ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
                else e0
        e_ = self.e_cell(self._select_parties(q_,qm_idx), e0)
        e_ = self.dropout(e_)
        return g_,q_,e_,alpha


class DialogueRNN(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e,
                            listener_state, context_attention, D_a, dropout)

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type())
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(U.type())
        e_ = torch.zeros(0).type(U.type())
        e = e_

        alpha = []
        for u_,qmask_ in zip(U, qmask):
            g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
            e = torch.cat([e, e_.unsqueeze(0)],0)
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        return e,alpha


class GRUModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(GRUModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
        
    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.gru(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions


class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):
        
        super(LSTMModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b, emotions

def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor

class DialogueGCNModel(nn.Module):

    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a,
                listener_state=False, context_attention='simple', dropout_rec=0.5, dropout=0.5,  avec=False,
                 no_cuda=False, graph_type='relation', use_topic=False, alpha=0.2, multiheads=6, graph_construct='direct', use_GCN=False,use_residue=True,
                 dynamic_edge_w=False,D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False, dataset='IEMOCAP',
                 use_speaker=True, use_modal=False, hyp_params=None):
        
        super(DialogueGCNModel, self).__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.multiheads = multiheads
        self.dropout = dropout
        self.use_GCN = use_GCN
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.dataset = dataset
        self.base_layer = 2
        self.av_using_lstm = False
        self.use_bert_seq = False
        self.hyp_params = hyp_params
        self.Multi_modle = MULTModel(self.hyp_params)

        if self.dataset == "IEMOCAP":
            input_size, hidden_size = 1024, 150
        else:
            input_size, hidden_size = 1024, 150

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=self.base_layer,
                           bidirectional=True, dropout=dropout)
        self.rnn_parties = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=self.base_layer,
                                   bidirectional=True, dropout=dropout)
        self.last_lstm = nn.LSTM(input_size=300, hidden_size=hidden_size, num_layers=2, bidirectional=True, dropout=0.0)


        if self.base_model == 'LSTM':
            if 'a' in self.modals:
                hidden_a = 300
                input_size_a = 1582
                hidden_size_a = 150
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                self.rnn_a = nn.LSTM(input_size=input_size_a, hidden_size=hidden_size_a, num_layers=self.base_layer,
                                   bidirectional=True, dropout=dropout)
                self.rnn_parties_a = nn.LSTM(input_size=input_size_a, hidden_size=hidden_size_a, num_layers=self.base_layer,
                                           bidirectional=True, dropout=dropout)
                # if self.av_using_lstm:
                #     self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'v' in self.modals:
                hidden_v = 300
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                input_size_v = 342
                hidden_size_v = 150
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                self.rnn_v = nn.LSTM(input_size=input_size_v, hidden_size=hidden_size_v, num_layers=self.base_layer,
                                   bidirectional=True, dropout=dropout)
                self.rnn_parties_v = nn.LSTM(input_size=input_size_v, hidden_size=hidden_size_v,
                                           num_layers=self.base_layer,
                                           bidirectional=True, dropout=dropout)
                # if self.av_using_lstm:
                #     self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'l' in self.modals:
                hidden_l = 200
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                else:
                    self.linear_l = nn.Linear(1024, hidden_l)
                self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=150, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2*D_e)

        else:
            print('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError


    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, lengths_dag=None):
        if 'a' in self.modals:
            U_a = self.linear_a(U_a)
            if self.av_using_lstm:
                emotions_a, hidden_a = self.lstm_a(U_a)
            else:
                emotions_a = U_a
                # emotions_a, hidden_a = self.last_lstm(U_a)
        if 'v' in self.modals:
            U_v = self.linear_v(U_v)
            if self.av_using_lstm:
                emotions_v, hidden_v = self.last_lstm(U_v)
            else:
                emotions_v = U_v
                # emotions_v, hidden_v = self.last_lstm(U_v)
        if 'l' in self.modals:
            if self.use_bert_seq:
                U_ = U.reshape(-1,U.shape[-2],U.shape[-1])
                U = self.txtCNN(U_).reshape(U.shape[0],U.shape[1],-1)
            else:
            #     U = self.linear_l(U)
            # emotions_l, hidden_l = self.lstm_l(U)
                if self.dataset == "IEMOCAP":
                    n_speakers = 2
                else:
                    n_speakers = 9
                U_s, U_p = None, None
                if self.base_model == 'LSTM':
                    # (b,l,h), (b,l,p)
                    U_, qmask_ = U, qmask.transpose(0, 1)
                    U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 300).type(U.type())
                    U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(n_speakers)]
                    for b in range(U_.size(0)):
                        for p in range(len(U_parties_)):
                            index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                            if index_i.size(0) > 0:
                                U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
                    E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in
                                  range(len(U_parties_))]

                    for b in range(U_p_.size(0)):
                        for p in range(len(U_parties_)):
                            index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                            if index_i.size(0) > 0:
                                U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                    U_p = U_p_.transpose(0, 1)

                    # (l,b,2*h) [(2*bi,b,h) * 2]
                    U_s, hidden = self.rnn(U)
                    U_s = U_s.transpose(0, 1)
                    emotions_l = U_s + U_p

                    emotions_l = emotions_l.transpose(1, 0)

        last_hs, out_put = self.Multi_modle(emotions_l, emotions_a, emotions_v, lengths_dag)
        return out_put

class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = 200, 200, 200
        # self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.d_l, self.d_a, self.d_v = 300, 300, 300
        self.vonly = True  # hyp_params.vonly
        self.aonly = True  # hyp_params.aonly
        self.lonly = True  # hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = 4
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.D_text, self.D_visual, self.D_audio = 300, 300, 300
        self.dataset = hyp_params.Dataset
        # self.D_text, self.D_visual, self.D_audio = 300, 342, 1582
        # 对输入的原始文本模态维度为300通过Linear变成600 方便后面与其他两种交互后的模态进行拼接
        self.l_extend = nn.Linear(self.D_text, 2 * self.d_l)

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        if hyp_params.Dataset == "IEMOCAP":
            output_dim = 6  # 8    # This is actually not a hyperparameter :-)
        else:
            output_dim = 7

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.D_text, self.d_l, kernel_size=1, padding=0, bias=False)  # (300,30) (100, 200)
        self.proj_a = nn.Conv1d(self.D_audio, self.d_a, kernel_size=1, padding=0, bias=False)  # (74,30) (1582, 200)
        self.proj_v = nn.Conv1d(self.D_visual, self.d_v, kernel_size=1, padding=0, bias=False)  # (35,30) (342, 200)
        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)  # a_mem
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)  # v_mem
        self.trans_last = self.get_network(self_type="last", layers=3)
        self.lstm_last = nn.LSTM(combined_dim, combined_dim)
        # Projection layers
        # self.proj1 = nn.Linear(self.D_text * 5, self.D_text * 5)
        self.proj2 = nn.Linear(self.D_text * 5, combined_dim)
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        # self.proj2 = nn.Linear(self.D_text * 5, self.D_text * 5)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        # self.out_layer = nn.Linear(self.D_text * 5, n_classes)

        # self.smax_fc = nn.linear(output_dim, hyp_params.n_classes)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout  # 2 * self.d_a
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'last':
            embed_dim, attn_dropout = 6 * self.d_a, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x_l, x_a, x_v, seq_lengths):
        """
        text should have dimension [batch_size, seq_len, n_features]
        """
        # x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_l = x_l.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        # 让 self.orig_d_l == d_l 不使用卷积操作进行特征维度的映射
        proj_x_l = x_l if self.D_text == self.d_l else self.proj_l(x_l)  # 将三种模态的数据进行映射
        proj_x_a = x_a if self.D_audio == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.D_visual == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # if self.lonly:
        #     # (V,A) --> L
        #     h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l) transformersEncoder
        #     h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L, N, d_l)
        #     h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        #     h_ls = self.trans_l_mem(h_ls)
        #     if type(h_ls) == tuple:
        #         h_ls = h_ls[0]
        #     # last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction 文本模态
        #     last_h_l = h_ls

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            # h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_a, proj_x_a)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            # h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_a, proj_x_a)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)  # h_as
            # h_as, hidden_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]
            last_h_a = h_as

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            # h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_v, proj_x_v)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)  # h_vs
            # h_vs, hidden_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
            last_h_v = h_vs

        if self.partial_mode == 3:
            # tensor 的维度为2时, dim =1  三维时修改dim = 2
            # last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=2)

            # Text main
            x_l = self.l_extend(x_l.transpose(1, 2))
            last_hs = torch.cat([x_l.permute(1, 0, 2), last_h_a, last_h_v], dim=2)  # 102

            # # # acoustic main
            # x_a = self.l_extend(x_a.transpose(1, 2))
            # last_hs = torch.cat([last_h_l, x_a.permute(1, 0, 2), last_h_v], dim=2)  # 102


            # # visual main
            # x_v = self.l_extend(x_v.transpose(1, 2))
            # last_hs = torch.cat([last_h_l, last_h_a, x_v.permute(1, 0, 2)], dim=2)  # 102

        # A residual block
        # last_hs_proj = self.proj1(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj = self.proj1(F.relu(self.proj1(last_hs)))

        # last_hs_proj = F.relu(self.proj1(last_hs))
        # last_hs_proj = self.proj1(last_hs)

        # last_hs_proj += last_hs

        # ### 尝试使用LSTM和Self-attention

        last_hs_proj = self.trans_last(last_hs_proj)


        output = self.out_layer(last_hs_proj)



        return last_hs_proj, output  # 交叉之后的结果，最后的预测输出  fusion ,preds