import sys

import pytorch_transformers as pt       # needs lukovnikov/master fork
import torch


def param(x:torch.Tensor):
    return torch.nn.Parameter(x.detach())


def get_bert(x:str="bert-base-uncased"):
    bert_tok = pt.BertTokenizer.from_pretrained(x)
    bert = pt.BertModel.from_pretrained(x, output_attentions=True, output_attentions_logits=True)
    return bert, bert_tok


def reduce_linear_dim(x:torch.nn.Linear, indim:int, outdim:int):
    x.weight = param(x.weight[:outdim, :indim])
    x.bias = param(x.bias[:outdim])
    x.in_features = indim
    x.out_features = outdim
    return x


def reduce_layernorm_dim(x, dim:int):
    x.weight = param(x.weight[:dim])
    x.bias = param(x.bias[:dim])
    return x


def reduce_embedding_dim(x:torch.nn.Embedding, dim:int):
    x.weight = param(x.weight[:, :dim])
    x.embedding_dim = dim
    return x


def make_lil_bert(bert:pt.BertModel, dim:int=420, vanilla=False):
    assert(float(int(420/bert.config.num_attention_heads)) == 420./bert.config.num_attention_heads)
    lil_bert = bert
    # lil embeddings
    lil_embs = bert.embeddings
    # print(lil_embs)
    reduce_embedding_dim(lil_embs.position_embeddings, dim)
    reduce_embedding_dim(lil_embs.token_type_embeddings, dim)
    reduce_embedding_dim(lil_embs.word_embeddings, dim)
    reduce_layernorm_dim(lil_embs.LayerNorm, dim)
    # lil pooler
    lil_pooler = bert.pooler.dense
    reduce_linear_dim(lil_pooler, indim=dim, outdim=dim)
    # lil encoder
    for lil_layer in lil_bert.encoder.layer:
        # print(lil_layer)
        lil_attention = lil_layer.attention.self
        reduce_linear_dim(lil_attention.key, indim=dim, outdim=dim)
        reduce_linear_dim(lil_attention.query, indim=dim, outdim=dim)
        reduce_linear_dim(lil_attention.value, indim=dim, outdim=dim)
        lil_attention.attention_head_size = int(dim / bert.config.num_attention_heads)
        lil_attention.all_head_size = lil_attention.num_attention_heads * lil_attention.attention_head_size

        lil_attention_output = lil_layer.attention.output
        reduce_linear_dim(lil_attention_output.dense, indim=dim, outdim=dim)
        reduce_layernorm_dim(lil_attention_output.LayerNorm, dim=dim)

        lil_interm = lil_layer.intermediate
        lil_inter_dim = int((lil_interm.dense.weight.size(0) / lil_interm.dense.weight.size(1)) * dim)
        reduce_linear_dim(lil_interm.dense, indim=dim, outdim=lil_inter_dim)

        lil_output = lil_layer.output
        reduce_linear_dim(lil_output.dense, indim=lil_inter_dim, outdim=dim)
        reduce_layernorm_dim(lil_output.LayerNorm, dim=dim)

        # print(lil_layer)
    # print(lil_bert)
    def reset_params(x):
        if hasattr(x, "reset_parameters"):
            x.reset_parameters()
    if vanilla:
        lil_bert.apply(reset_params)
    return lil_bert


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

    def forward(self, student_attention_logits, teacher_attention_logits):
        """
        :param teacher_attention_logits:    (batsize, numlayers, numheads, seqlen, seqlen)
        :param student_attention_logits:    (batsize, numlayers, numheads, seqlen, seqlen)
        :return:
        """
        distance = (teacher_attention_logits - student_attention_logits).norm(2, -1)
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
        # feed through teacher, get attention logits and output logits
        with torch.no_grad():
            teacher_logits, teacher_attention_logits = self.teacher(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids)
            teacher_attention_logits = torch.stack(teacher_attention_logits, 1)
        # attention logits must be (batsize, numlayers, numheads, seqlen, seqlen)
        # feed through student, get same
        student_logits, student_attention_logits = self.student(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids)
        student_attention_logits = torch.stack(student_attention_logits, 1)
        # compute loss
        loss = self.get_loss(targets, student_logits, teacher_logits, student_attention_logits, teacher_attention_logits)
        return loss, student_logits

    def get_loss(self, g, pred, teacher_pred, sa_logits, ta_logits):
        return self.loss(pred, g, teacher_pred)


class BertDistillWithAttentionModel(BertDistillModel):
    def __init__(self, teacher, student, alpha=.5, beta=1., **kw):
        super(BertDistillWithAttentionModel, self).__init__(teacher, student, alpha=alpha, **kw)
        self.beta = beta
        self.att_loss = AttentionDistillLoss()

    def get_loss(self, g, pred, teacher_pred, sa_logits, ta_logits):
        mainl = super(BertDistillWithAttentionModel, self).get_loss(g, pred, teacher_pred, sa_logits, ta_logits)
        attl = self.att_loss(sa_logits, ta_logits)
        ret = mainl + self.beta * attl
        return ret


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
    teacher, tok = get_bert()
    student, _ = get_bert()
    student = make_lil_bert(student, dim=420, vanilla=True)
    teacher = BertClassifier(teacher, 768, 5)
    student = BertClassifier(student, 420, 5)
    m = BertDistillWithAttentionModel(teacher, student, alpha=.5, beta=1.)

    x = "lil bert went for a walk"
    xtok = torch.tensor(tok.encode(x)).unsqueeze(0)
    y = m(xtok, targets=torch.randint(0, 4, (1,)))

    l = y[0]
    l.backward()
    print(student.bert.embeddings.word_embeddings.weight.grad[:, 0].nonzero())


if __name__ == '__main__':
    try_bert_distill_model_with_attention()
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