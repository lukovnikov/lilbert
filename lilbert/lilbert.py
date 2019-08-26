import copy
import math
import sys
from functools import partial
from typing import Union

import pytorch_transformers as pt       # needs lukovnikov/master fork
import torch
import numpy as np


DEBUG = True


def param(x:torch.Tensor):
    return torch.nn.Parameter(x.detach())


def get_bert(x:str="bert-base-uncased"):
    bert_tok = pt.BertTokenizer.from_pretrained(x)
    bert = pt.BertModel.from_pretrained(x, output_attentions=True, output_attentions_logits=True)
    return bert, bert_tok


def reduce_linear_dim(x:torch.nn.Linear, indim:int, outdim:int):
    x.weight = param(x.weight.detach()[:outdim, :indim])
    x.bias = param(x.bias.detach()[:outdim])
    x.in_features = indim
    x.out_features = outdim
    return x


def reduce_projection_dim(x:torch.nn.Linear, indim:int, outdim:int, numheads:int):
    """ x is a projection layer """
    # do weight
    w = x.weight.detach()
    ws = torch.chunk(w, numheads, 0)
    ws = [we[:outdim//numheads, :indim] for we in ws]
    w = torch.cat(ws, 0)
    x.weight = param(w)
    # do bias
    b = x.bias.view(numheads, int(x.bias.size(0)/numheads))
    b = b.detach()[:, :int(outdim/numheads)]
    b = b.contiguous().view(outdim)
    x.bias = param(b)
    # dims
    x.in_features = indim
    x.out_features = outdim
    return x


def reduce_linear_dim_headstriped_input(x:torch.nn.Linear, indim:int, outdim:int, numheads:int):
    # do weight
    w = x.weight.detach()
    ws = torch.chunk(w, numheads, 1)    # chunk by input
    ws = [we[:outdim, :indim//numheads] for we in ws]
    w = torch.cat(ws, 1)
    x.weight = param(w)
    # normal reduction:
    x.bias = param(x.bias.detach()[:outdim])
    x.in_features = indim
    x.out_features = outdim
    return x


def reduce_layernorm_dim(x, dim:int):
    x.weight = param(x.weight.detach()[:dim])
    x.bias = param(x.bias.detach()[:dim])
    return x


def reduce_embedding_dim(x:torch.nn.Embedding, dim:int):
    x.weight = param(x.weight.detach()[:, :dim])
    x.embedding_dim = dim
    return x


class ClassificationDistillLoss(torch.nn.Module):
    def __init__(self, alpha=.5, reduction="mean", **kw):
        """
        :param alpha:       mixing between hard CE and distill. alpha == 1 => only hard CE loss
        :param kw:
        """
        super(ClassificationDistillLoss, self).__init__(**kw)
        self.alpha = alpha
        self.reduction = reduction
        assert(self.reduction == "mean")
        self.CEloss = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits, targets, targetlogits):
        """
        :param logits:       (batsize, numclasses) unnormalized logits from classifier
        :param targets:      (batsize, ) int id of correct class
        :param targetlogits: (batsize, numclasses) unnormalized logits from parent classifier
        :return:
        """
        celoss = self.CEloss(logits, targets)
        distance = 0
        if self.alpha < 1:
            distance = (logits - targetlogits).norm(2, 1) ** 2
            distance = distance.mean() if self.reduction == "mean" else distance.sum()
        ret = celoss * self.alpha + (1 - self.alpha) * distance
        return ret


def try_classification_distill_loss():
    logits = param(torch.rand(5, 4))
    targets = torch.randint(0, 4, (5,))
    targetlogits = param(torch.rand(5, 4))
    loss = ClassificationDistillLoss(alpha=.5)
    l = loss(logits, targets, targetlogits)
    print(l)
    l.backward()
    print(logits.grad)


class AttentionDistillLoss(torch.nn.Module):
    def __init__(self, reduction="mean", **kw):
        super(AttentionDistillLoss, self).__init__(**kw)
        self.reduction = reduction

    def forward(self, student_attention_logits, teacher_attention_logits, attention_mask):
        """
        :param teacher_attention_logits:    (batsize, numlayers, numheads, seqlen, seqlen)
        :param student_attention_logits:    (batsize, numlayers, numheads, seqlen, seqlen)
        :param attention_mask:              (batsize, seqlen)
        :return:
        """
        att_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)
        distances = (teacher_attention_logits - student_attention_logits)
        distances = distances * att_mask.float()
        distance = distances.norm(2, -1) ** 2 #(distances ** 2).sum(-1))     # (batsize, numl, numh, seqlen)
        # average over sequence elements in example
        att_mask = att_mask.sum(-1) > 0
        distance = distance.sum(-1) / att_mask.float().sum(-1)
        # average over heads and layers
        distance = distance.mean(-1).mean(-1)
        # distance = distances.norm(2, -1)
        # average over examples
        distance = distance.mean() if self.reduction == "mean" else distance.sum()
        return distance


def try_attention_distill_loss():
    sa_logits = param(torch.rand(5, 12, 12, 6, 6))
    ta_logits = param(torch.rand(5, 12, 12, 6, 6))
    loss = AttentionDistillLoss()
    l = loss(sa_logits, ta_logits)
    print(l)
    l.backward()
    print(sa_logits.grad.size(), sa_logits.grad.norm())


class BertClassifier(torch.nn.Module):      # almost straight from pytorch-transformers
    def __init__(self, bert, dim, numclasses, dropout=0., **kw):
        super(BertClassifier, self).__init__(**kw)

        self.bert = bert
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(dim, numclasses)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = input_ids != 0
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        attentions = outputs[2]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, attentions


class BertClassifierModel(torch.nn.Module):
    def __init__(self, bertclassifier:BertClassifier, **kw):
        super(BertClassifierModel, self).__init__(**kw)
        self.m = bertclassifier
        self.loss = ClassificationDistillLoss(alpha=1.)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None, targets=None):
        logits, _ = self.m(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                           position_ids=position_ids, head_mask=head_mask)
        l = self.loss(logits, targets)
        return l, logits


class BertDistillModel(torch.nn.Module):
    """
    Model for normal BERT distillation onto a lil BERT
    """

    def __init__(self, teacher, student, alpha=.5, **kw):
        """
        :param teacher: normal BERT with classifier on top
        :param student: lil BERT with clasifier on top
        :param kw:
        """
        super(BertDistillModel, self).__init__(**kw)
        self.teacher, self.student = teacher, student
        self.loss = ClassificationDistillLoss(alpha=alpha)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,
                targets=None):
        """
        Same as BertModel
        :param targets: (batsize,) classification target int ids
        """
        if attention_mask is None:
            attention_mask = input_ids != 0
        # feed through teacher, get attention logits and output logits
        with torch.no_grad():
            teacher_logits, teacher_attention_logits = self.teacher(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids)
            teacher_attention_logits = torch.stack(teacher_attention_logits, 1)
        # attention logits must be (batsize, numlayers, numheads, seqlen, seqlen)
        # feed through student, get same
        student_logits, student_attention_logits = self.student(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids)
        student_attention_logits = torch.stack(student_attention_logits, 1)
        # compute loss
        loss = self.get_loss(targets, student_logits, teacher_logits, student_attention_logits, teacher_attention_logits, attention_mask)
        return loss, student_logits

    def get_loss(self, g, pred, teacher_pred, sa_logits, ta_logits, att_mask):
        return self.loss(pred, g, teacher_pred)


class BertDistillWithAttentionModel(BertDistillModel):
    def __init__(self, teacher, student, alpha=.5, beta=1., **kw):
        super(BertDistillWithAttentionModel, self).__init__(teacher, student, alpha=alpha, **kw)
        self.beta = beta
        self.att_loss = AttentionDistillLoss()

    def get_loss(self, g, pred, teacher_pred, sa_logits, ta_logits, att_mask):
        mainl = super(BertDistillWithAttentionModel, self).get_loss(g, pred, teacher_pred, sa_logits, ta_logits, att_mask)
        attl = self.att_loss(sa_logits, ta_logits, att_mask)
        ret = mainl + self.beta * attl
        return ret


class PrunedLinear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, linear:torch.nn.Linear):
        super(PrunedLinear, self).__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight, self.bias = linear.weight, linear.bias
        self.W_mask = None
        self._debug = DEBUG

    def prune_magnitude(self, frac):
        vals, ids = torch.sort(self.weight.view(-1).abs(), descending=True)
        numretained = int(frac * len(vals))
        cutoff_value = vals[numretained]
        self.W_mask = self.weight.abs() > cutoff_value
        self.weight = torch.nn.Parameter((self.weight * self.W_mask.float()).detach())
        if self._debug:
            numleft = self.W_mask.sum().float().detach().cpu().item()
            print(f"Number of elements left: {numleft/(self.weight.size(0)*self.weight.size(1))}% {numleft} ({numretained}) with cutoff {cutoff_value} for frac={frac}")

    def forward(self, input):
        weight = self.weight * self.W_mask.float() if self.W_mask is not None else self.weight
        ret = torch.nn.functional.linear(input, weight, self.bias)
        return ret

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class PrunedEmbedding(torch.nn.Embedding):
    def __init__(self, emb:torch.nn.Embedding):
        super(PrunedEmbedding, self).__init__(emb.num_embeddings, emb.embedding_dim)
        self.padding_idx = emb.padding_idx
        self.max_norm = emb.max_norm
        self.norm_type = emb.norm_type
        self.scale_grad_by_freq = emb.scale_grad_by_freq
        self.weight = torch.nn.Parameter(emb.weight.detach())
        self.sparse = emb.sparse
        self.W_mask = None
        self._debug = DEBUG

    def prune_magnitude(self, frac):
        vals, ids = torch.sort(self.weight.abs(), descending=True, dim=1)
        numretained = int(frac * vals.size(1))
        cutoff_values = vals[:, numretained]
        self.W_mask = self.weight.abs() > (cutoff_values.unsqueeze(1))
        self.weight = torch.nn.Parameter((self.weight * self.W_mask.float()).detach())
        if self._debug:
            numleft = self.W_mask.sum().float().detach().cpu().item()
            print(f"Number of elements left: {numleft/(self.weight.size(0)*self.weight.size(1))}% {numleft} ({numretained}) for frac={frac}")

    def forward(self, input):
        weight = self.weight * self.W_mask.float() if self.W_mask is not None else self.weight
        ret = torch.nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return ret


def try_prune_emb():
    l = torch.nn.Embedding(1000, 10)
    print(l.weight[:10])
    pl = PrunedEmbedding(l)
    pl.prune_magnitude(.1)
    print(pl.weight[:10])


def try_prune_linear():
    l = torch.nn.Linear(10,10)
    print(l.weight)
    pl = PrunedLinear(l)
    pl.prune_magnitude(.5)
    print(pl.weight)
    y = pl(torch.rand(2, 10))
    loss = y.sum()
    loss.backward()
    print(pl.weight.grad)


def prune_linear_submodules(fraction:float, m:torch.nn.Module):
    repl_dict = {}
    for subname, submodule in m.named_children():
        if isinstance(submodule, torch.nn.Linear):
            replacement = PrunedLinear(submodule)
            replacement.prune_magnitude(fraction)
            repl_dict[subname] = replacement
    for k, v in repl_dict.items():
        setattr(m, k, v)


def prune_embedding_submodules(fraction:float, m:torch.nn.Module):
    repl_dict = {}
    for subname, submodule in m.named_children():
        if isinstance(submodule, torch.nn.Embedding):
            replacement = PrunedEmbedding(submodule)
            replacement.prune_magnitude(fraction)
            repl_dict[subname] = replacement
    for k, v in repl_dict.items():
        setattr(m, k, v)


def make_lil_bert_prune(bert: pt.BertModel, fraction=0.3):
    _bert = copy.deepcopy(bert)
    prune_emb_f = partial(prune_embedding_submodules, fraction)
    _bert.bert.apply(prune_emb_f)
    prune_linear_f = partial(prune_linear_submodules, fraction)
    _bert.bert.apply(prune_linear_f)
    return _bert


def try_prune_lil_bert():
    teacher, tok = get_bert()
    teacher = BertClassifier(teacher, 768, 5)
    student = make_lil_bert(teacher, fraction=0.1, method="prune")
    m = BertDistillModel(teacher, student, alpha=.5)
    # print(student)
    print(student.bert.encoder.layer[5].attention.self.query.weight)




def make_lil_bert(bert: Union[pt.BertModel, BertClassifier], dim: int = 420, vanilla=True, vanilla_emb=False, fraction:float=0.3, method="cut"):
    if method == "cut":
        print("using method CUT")
        return make_lil_bert_cut(bert, dim=dim, vanilla=vanilla, vanilla_emb=vanilla_emb)
    elif method == "prune":
        print("using pruning by magnitude")
        return make_lil_bert_prune(bert=bert, fraction=fraction)
    else:
        raise Exception(f"Unknown method {method}")


def make_lil_bert_cut(bert: Union[pt.BertModel, BertClassifier], dim: int = 420, vanilla=True, vanilla_emb=False):
    def reset_params(x):
        if hasattr(x, "reset_parameters"):
            x.reset_parameters()

    _bert_ret = copy.deepcopy(bert)
    if isinstance(_bert_ret, BertClassifier):
        # cut the output layer of classifier too
        reduce_linear_dim(_bert_ret.classifier, dim, _bert_ret.classifier.out_features)
        if vanilla:
            _bert_ret.classifier.apply(reset_params)
        _bert = _bert_ret.bert
    else:
        _bert = _bert_ret

    assert (float(int(420 / _bert.config.num_attention_heads)) == 420. / _bert.config.num_attention_heads)
    lil_bert = _bert
    # lil embeddings
    lil_embs = _bert.embeddings
    # print(lil_embs)
    reduce_embedding_dim(lil_embs.position_embeddings, dim)
    reduce_embedding_dim(lil_embs.token_type_embeddings, dim)
    reduce_embedding_dim(lil_embs.word_embeddings, dim)
    reduce_layernorm_dim(lil_embs.LayerNorm, dim)
    # lil pooler
    lil_pooler = _bert.pooler.dense
    reduce_linear_dim(lil_pooler, indim=dim, outdim=dim)
    # lil encoder
    for lil_layer in lil_bert.encoder.layer:
        # print(lil_layer)
        lil_attention = lil_layer.attention.self
        reduce_projection_dim(lil_attention.key, indim=dim, outdim=dim, numheads=lil_attention.num_attention_heads)
        reduce_projection_dim(lil_attention.query, indim=dim, outdim=dim, numheads=lil_attention.num_attention_heads)
        reduce_projection_dim(lil_attention.value, indim=dim, outdim=dim, numheads=lil_attention.num_attention_heads)
        lil_attention.attention_head_size = int(dim / _bert.config.num_attention_heads)
        lil_attention.all_head_size = lil_attention.num_attention_heads * lil_attention.attention_head_size

        lil_attention_output = lil_layer.attention.output
        reduce_linear_dim_headstriped_input(lil_attention_output.dense, indim=dim, outdim=dim,
                                            numheads=lil_attention.num_attention_heads)
        reduce_layernorm_dim(lil_attention_output.LayerNorm, dim=dim)

        lil_interm = lil_layer.intermediate
        lil_inter_dim = int((lil_interm.dense.weight.size(0) / lil_interm.dense.weight.size(1)) * dim)
        reduce_linear_dim(lil_interm.dense, indim=dim, outdim=lil_inter_dim)

        lil_output = lil_layer.output
        reduce_linear_dim(lil_output.dense, indim=lil_inter_dim, outdim=dim)
        reduce_layernorm_dim(lil_output.LayerNorm, dim=dim)

        # print(lil_layer)

    # print(lil_bert)

    if vanilla:
        lil_pooler.apply(reset_params)
        lil_bert.encoder.apply(reset_params)
    if vanilla_emb:
        lil_embs.apply(reset_params)
    return _bert_ret


def try_bert_distill_model():
    teacher, tok = get_bert()
    student, _ = get_bert()
    student = make_lil_bert(student, dim=420, vanilla=True)
    teacher = BertClassifier(teacher, 768, 5)
    student = BertClassifier(student, 420, 5)
    m = BertDistillModel(teacher, student, alpha=.5)

    x = "lil bert went for a walk"
    xtok = torch.tensor(tok.encode(x)).unsqueeze(0)
    y = m(xtok, targets=torch.randint(0, 4, (1,)))

    l = y[0]
    l.backward()
    print(student.bert.embeddings.word_embeddings.weight.grad[:, 0].nonzero())


def try_bert_distill_model_with_attention():
    teacher_, tok = get_bert()
    # student_ = make_lil_bert(teacher_, dim=420, vanilla=False)
    teacher = BertClassifier(teacher_, 768, 5)
    # student = BertClassifier(student_, 420, 5)
    student = make_lil_bert(teacher, 420)
    student_ = student.bert

    print(student.bert.embeddings.word_embeddings.weight.size())
    print(student.bert.encoder.layer[5])
    print(student.bert.encoder.layer[5].attention.self.query.weight.size())

    m = BertDistillWithAttentionModel(teacher, student, alpha=.5, beta=.5)

    xs = ["lil bert went for a walk", "he found ernie [PAD] [PAD] [PAD]"]
    xtoks = [torch.tensor(tok.encode(x)).unsqueeze(0) for x in xs]
    y = m(torch.cat(xtoks, 0), targets=torch.randint(0, 4, (2,)))

    teacher_wordembs = teacher_.embeddings.word_embeddings.weight.detach().numpy().copy()+0
    student_wordembs = student_.embeddings.word_embeddings.weight.detach().numpy().copy() + 0
    print(teacher_wordembs[1037, :5])
    print(student_wordembs[1037, :5])

    l = y[0]
    l.backward()
    optim = torch.optim.SGD(m.parameters(), lr=1)
    # optim.zero_grad()
    print(student.bert.embeddings.word_embeddings.weight.grad[:, 0].nonzero())
    print(teacher_.embeddings.word_embeddings.weight.grad)
    assert(teacher_.embeddings.word_embeddings.weight.grad is None)
    assert(student_.embeddings.word_embeddings.weight.grad is not None)
    optim.step()

    teacher_wordembs_after = teacher_.embeddings.word_embeddings.weight.detach().numpy().copy()+0
    student_wordembs_after = student_.embeddings.word_embeddings.weight.detach().numpy().copy() + 0
    print(teacher_wordembs_after[1037,:5])
    print(student_wordembs_after[1037,:5])
    assert(np.allclose(teacher_wordembs, teacher_wordembs_after))
    assert(not np.allclose(student_wordembs, student_wordembs_after))
    print("DONE.")


def try_reduce_projection_dim():
    layer = torch.nn.Linear(768, 768)
    x = torch.ones(1,768)
    y = layer(x)
    reduce_projection_dim(layer,768,420,12)
    xr = x[:, :420]
    yr = layer(x)
    print(yr[:, 420//12:420//12+768//12] - y[:, 768//12:768//6])


def try_reduce_project_dim_selfattnlayer():
    bert, tok = get_bert()
    l = bert.encoder.layer[0].attention.self
    lil_l = copy.deepcopy(l)
    reduce_projection_dim(lil_l.query, 768, 420, 12)
    reduce_projection_dim(lil_l.key, 768, 420, 12)
    reduce_projection_dim(lil_l.value, 768, 420, 12)
    lil_l.attention_head_size = 420//12
    lil_l.all_head_size = 420
    x = torch.randn(1,2,768)
    am = torch.ones(1, 1, 2, 2)
    y = l(x, am)[0]
    yr = lil_l(x, am)[0]
    print(yr[:, :, 420 // 12:420 // 12 + 768 // 12] - y[:, :, 768 // 12:768 // 6])


if __name__ == '__main__':
    # try_reduce_projection_dim()
    # sys.exit()
    # try_reduce_project_dim_selfattnlayer()
    # sys.exit()
    # try_prune_linear()
    try_prune_lil_bert()
    sys.exit()
    try_bert_distill_model_with_attention()
    sys.exit()
    try_bert_distill_model()
    try_classification_distill_loss()
    try_attention_distill_loss()
    sys.exit()

    bert, bert_tok = get_bert()
    # test run
    x = "lil bert gucci gang gucci gang"
    xtok = torch.tensor(bert_tok.encode(x)).unsqueeze(0)
    y = bert(xtok)
    last_state, yyy, attentions = y
    print(attentions[0].size())

    lil_bert = make_lil_bert(bert)
    lily = lil_bert(xtok)
    print(lily[2][0].size())

    print(attentions[0])
    print(lily[2][0])