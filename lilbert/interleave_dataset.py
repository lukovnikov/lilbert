from __future__ import absolute_import, division, print_function

import sys

sys.path.append("../transformers/")
sys.path.append("../lilbert")

import os
import torch
import wandb
import random
import logging
import argparse
import numpy as np
from typing import Union
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

# Local imports
import lilbert
from typing import Union, Any, Tuple, List
from utils import goodies as gd
from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors, GLUE_TASKS_NUM_LABELS)

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class ProbIter:

    def __init__(self, length, method='lin'):
        # TODO: implement next with a bunch of asserts.
        # TODO: Generalize for multiple datasets at a go.
        self.len, self.method = length, method
        if self.method == 'lin':
            data_a = np.arange(0, self.len+1, 1) / float(self.len)
            data_b = np.arange(self.len, -1, -1) / float(self.len)
            self.data = zip(data_a, data_b)

    def __len__(self):
        return self.len

    def __iter__(self):
        # If logic needed in _next_, return self, and write __next__(self)
        self.i = 0
        return self

    def __next__(self):
        return self.data.__next__()


class InterpolatingIter:

    def __init__(self, data_a: Tuple[np.ndarray], data_b: Tuple[np.ndarray], bs: int = 32, prob_method: str = 'lin' ):
        """
            Take two datasets, shuffle them. Store them.
            Find Length of this iter
            Make a prob iter correspondingly

        :param data_a:
        :param data_b:

        """
        self.bs = bs
        self.data = [data_a, data_b]
        self.data_nm = [0, 1]
        # self.data = [(index, self._shuffle_(d)) for index, d in enumerate(self.data)]
        self._shuffle_()
        self.len = sum(len(d[0])//self.bs  - (1 if len(d[0]) % self.bs else 0) for d in self.data)
        self.probs = ProbIter(self.len, prob_method)

    def _shuffle_(self):
        for i in range(len(self.data)):
            data = self.data[i]
            index = np.arange(len(data[0]))
            np.random.shuffle(index)
            self.data[i] = tuple([datum[index] for datum in data])

    def __len__(self) -> int:
        return self.len

    def __iter__(self):
        self.i, self.iter = [0 for _ in self.data], 0
        return self

    def reset(self):
        self.i, self.iter = [0 for _ in self.data], 0
        self._shuffle_()

    def _return_data_(self, src, pos):
        """ Actual Data pulling happens here """

        if pos+self.bs < len(src[0]):           # No Edge Case
            data_chunk = [datum[pos:pos+self.bs] for datum in src]
            pos += self.bs
        else:                                   # Edge Case
            data_chunk = [datum[pos:] for datum in src]
            pos = 0
        return data_chunk, pos


    def __next__(self) -> (int, Any):

        # Length check
        if self.iter >= self.len:
            raise StopIteration
        else:
            self.iter += 1

        index= np.random.choice(self.data_nm, p=self.probs.__next__())
        data_src = self.data[index]
        data_chunk, pos = self._return_data_(data_src, self.i[index])
        self.i[index] = pos

        return index, data_chunk


def train(args: Union[dict, gd.FancyDict], train_dataset, train_dataset_aux, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    #
    # train_sampler_aux = RandomSampler(train_dataset_aux) if args.local_rank == -1 else DistributedSampler(train_dataset_aux)
    # train_dataloader_aux = DataLoader(train_dataset_aux, sampler=train_sampler_aux, batch_size=args.train_batch_size)

    train_dataloader = InterpolatingIter(train_dataset.tensors, train_dataset_aux.tensors,
                                         bs=args.per_gpu_train_batch_size, prob_method='lin')

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.call_wandb:
        wandb.config['Total optimization steps'] = t_total

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    global_step_aux = 0

    tr_loss, logging_loss = 0.0, 0.0
    tr_loss_aux, logging_loss_aux = 0.0, 0.0

    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])


    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    previous_accuracy, previous_accuracy_aux = 0, 0

    # interleave_iter = InterpolatingIter()

    for i in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # epoch_iterator_aux = tqdm(train_dataloader_aux, desc="Iteration", disable=args.local_rank not in [-1, 0])

        for step, (datasetnm, batch) in enumerate(epoch_iterator):


            if datasetnm == 0:
                logging_loss, global_step, tr_loss = forward_pass(args, model, batch, tr_loss, global_step, step,
                                                                  scheduler, optimizer, previous_accuracy, logging_loss
                                                                  , tokenizer, 0.5, do_evaluate=True)
            elif datasetnm == 1:
                logging_loss_aux, global_step_aux, tr_loss_aux  = \
                        forward_pass(args, model, batch, tr_loss_aux, global_step_aux, step, scheduler, optimizer,
                                           previous_accuracy_aux, logging_loss_aux
                                 , tokenizer, 1.0, do_evaluate=False)

            else:
                raise IOError

        train_dataloader.reset()

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.call_wandb:
        wandb.config['global_step'] = global_step
        wandb.config['global_loss'] = tr_loss / global_step
    return global_step, tr_loss / global_step

def forward_pass(args, model, batch, tr_loss, global_step, step, scheduler, optimizer, previous_accuracy, logging_loss
                 , tokenizer, custom_alpha, do_evaluate=True):
    model.train()
    batch = tuple(t.to(args.device) for t in batch)
    if args.mode == 'loss_in_train_loop':
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None}

        outputs = model(**inputs)
        logits = outputs[0]
        inputs['labels'] = batch[3]

        # outputs = (logits,) + outputs[2:]  # add hidden hiddenstates and attention if they are here

        # Calculating loss
        if inputs['labels'] is not None:
            if args.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), inputs['labels'].view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, args.num_labels), inputs['labels'].view(-1))
            outputs = (loss,) + outputs

        loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

    elif args.mode == 'loss_in_model':
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                  'targets': batch[3],
                  'custom_alpha':custom_alpha }

        loss, logits = model(**inputs)
        inputs['labels'] = batch[3]

    else:
        print(f"mode not recognized. mode found {args.mode}")
        raise IOError

    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps


    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    iter_loss = loss.item()
    tr_loss += iter_loss

    # logging loss for wandb
    if global_step % args.logging_loss_steps == 0 and args.call_wandb:
        # log the loss here
        wandb.log({'iter_loss': iter_loss})

    if (step + 1) % args.gradient_accumulation_steps == 0:
        scheduler.step()  # Update learning rate schedule
        optimizer.step()
        if args.pruner:
            args.pruner.step()
        model.zero_grad()
        global_step += 1

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and\
                global_step % args.logging_steps == 0 and do_evaluate:
            # Log metrics
            if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                results = evaluate(args, model, tokenizer)

                if args.call_wandb:
                    wandb.log({k: v for k, v in results.items()})

                for k, v in results.items():
                    if k == 'acc':
                        key = 'acc'
                    elif k == 'mcc':
                        key = 'mcc'
                    elif k == 'corr':
                        key = 'corr'
                    elif k == 'f1':
                        key = 'f1'
                    elif k == 'acc_and_f1':
                        key = 'acc_and_f1'
                    elif k == 'pearson':
                        key = 'pearson'
                    elif k == 'spearmanr':
                        key = 'spearmanr'
                    else:
                        raise gd.UnknownAccuracyMetric(f"The current training loop only"
                                                       f" supports acc, mcc, corr, acc_and_f1, f1, pearson,"
                                                       f" and spearmanr"
                                                       f". Found {k}")

                if previous_accuracy < results[key]:  # acc, mcc, corr.
                    # Note that previous accuracy could be acc, mrr, corr
                    previous_accuracy = results[key]
                    if args.call_wandb:
                        wandb.log({'best_acc': previous_accuracy})
                    # save the model here
                    if args.save:
                        gd.save_model(model=model, output_dir=args.output_dir,
                                      model_name=args.task_name + args.output_name, accuracy=results[key],
                                      config={"mode": args.mode})

                # for key, value in results.items():
                #     tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
            if args.call_wandb:
                wandb.log({'lr': scheduler.get_lr()[0]})
                wandb.log({'loss': tr_loss - logging_loss})

            # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            # tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
            logging.info(f"the current training loss is {tr_loss - logging_loss}")
            logging_loss = tr_loss

    return logging_loss, global_step, tr_loss

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                if args.mode == 'loss_in_train_loop':
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              }
                    outputs = model(**inputs)
                    logits = outputs[0]
                    inputs['labels'] = batch[3]

                    # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

                    # Calculating loss
                    if inputs['labels'] is not None:
                        if args.num_labels == 1:
                            #  We are doing regression
                            loss_fct = MSELoss()
                            loss = loss_fct(logits.view(-1), inputs['labels'].view(-1))
                        else:
                            loss_fct = CrossEntropyLoss()
                            loss = loss_fct(logits.view(-1, args.num_labels), inputs['labels'].view(-1))
                        outputs = (loss,) + outputs

                    tmp_eval_loss, logits = outputs[:2]
                    eval_loss += tmp_eval_loss.mean().item()
                else:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              'targets': batch[3]}
                    loss, logits = model(**inputs)
                    inputs['labels'] = batch[3]

                    # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

                    eval_loss += loss.mean().item()

            if args.call_wandb:
                wandb.log({'eval_loss': eval_loss})

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, trim=1.0):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    if trim < 1:
        index = np.random.choice(np.arange(len(all_input_ids)), size=int(len(all_input_ids)*trim))
        dataset = TensorDataset(
            all_input_ids[index], all_input_mask[index], all_segment_ids[index], all_label_ids[index]
        )
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main(args, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        logger.info(f"dataset chossen is {args.task_name}")

        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)

        dd = args.data_dir
        d = args.task_name

        args.task_name = 'qqp'
        args.data_dir = 'dataset/QQP'
        train_dataset_aux = load_and_cache_examples(args, 'qqp', tokenizer, evaluate=False, trim=0.03)

        args.data_dir = dd
        args.task_name = d


        logger.info("About to begin training")
        global_step, tr_loss = train(args=args, train_dataset=train_dataset, train_dataset_aux = train_dataset_aux,
                                     model=model, tokenizer=tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == '__main__':

    '''
        save - if True:
                    saves the model

        mode 
            - loss_in_train_loop
            - loss_in_model

        # in BERTClassifier the loss is calculated in the train loop
        # in BertDistillWithAttentionModel and BertDistill loss is calculated in the model and not training loop.

    '''

    args = gd.FancyDict(
        notes='',
        mode='loss_in_model',  # loss_in_train_loop, loss_in_model
        from_scratch=False,  # False, True
        method='cut',  # 'prune','cut','linear_pruning'
        loss_type='attention',  # attention, distill
        teacher_dim=768,  # default for small BERT
        student_dim=420,  # smaller (~35% params of small BERT)
        save=False,
        only_teacher=False,
        t_total=10000,
        n_steps=100,
        data_dir='dataset/RTE',
        model_type='bert',
        model_name='bert-base-cased',
        model_name_or_path='bert-base-uncased',
        task_name='RTE',
        output_dir='output/bert/',
        output_name='_model.pt',
        config_name='bert-base-cased',  # change according to model name. Should be same as model name.
        tokenizer_name='BertTokenizer',
        cache_dir='',
        max_seq_length=128,
        do_train=True,
        do_eval=True,
        evaluate_during_training=True,
        do_lower_case=True,
        per_gpu_train_batch_size=32,
        per_gpu_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=6.0,
        max_steps=-1,
        warmup_steps=0,
        logging_steps=1000,
        logging_loss_steps=10,
        save_steps=1000,
        eval_all_checkpoints=True,
        no_cuda=True,
        overwrite_output_dir=True,
        overwrite_cache=True,
        seed=42,
        fp16=False,
        fp16_opt_level=1,
        local_rank=-1,
        server_ip='',
        server_port='',
        pruner=None,
        call_wandb=False,
        alpha=0.5,
        prune_frac=0.1,
        prune_frac_emb=0.5)

    # Diff in args
    args.call_wandb = False
    args.mode = 'loss_in_model'

    args.method = 'cut'
    args.loss_type = 'attention'  # attention
    args.only_teacher = False
    args.save = False
    args.alpha = 0.5
    args.data_dir = 'dataset/MRPC'
    args.task_name = 'MRPC'

    # various experiment varioation

    # TODO: REPLACE!!!!
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="increase output verbosity")
    cmd_args = parser.parse_args()
    if cmd_args.dataset:
        args.task_name = cmd_args.dataset
        args.data_dir = 'dataset/' + args.task_name

    # dataset size
    dataset_logging = {
        'CoLA': 100,
        'SST-2': 1000,
        'MRPC': 30,
        'STS-B': 100,
        'QQP': 1000,
        'MNLI': 1000,
        'QNLI': 100,
        'RTE': 10,
        'WNLI': 10
    }

    args.logging_loss_steps = dataset_logging[args.task_name]
    args.logging_steps = dataset_logging[args.task_name]

    if args.save:
        assert args.only_teacher is True and args.mode == 'loss_in_train_loop', "the codebase only " \
                                                                                "supports saving teacher. To train " \
                                                                                "teacher set only_teacher args to true"
    # number of classes for a given dataset
    args.numclasses = GLUE_TASKS_NUM_LABELS[args.task_name.lower()]

    if args.mode == 'loss_in_train_loop':

        if args.only_teacher:
            # Snippet to make a bert classifier with given dim -1
            teacher, tok = lilbert.get_bert()
            model = lilbert.BertClassifier(teacher, args.teacher_dim, args.numclasses)

        else:
            # Snippet to make a bert classifier with lesser dim - 5,6
            teacher, tok = lilbert.get_bert()
            student = lilbert.make_lil_bert(teacher.bert, dim=args.student_dim, method="cut",
                                            vanilla=args.from_scratch)  # This can TAKE BOTH TEACHER AS WELL AS TEACHER.BERT
            model = lilbert.BertClassifier(student, args.student_dim, args.numclasses)

    elif args.mode == 'loss_in_model':

        # Lilbert with attention distillation - 2a
        _, tok = lilbert.get_bert()
        device = torch.device('cpu')
        teacher = torch.load(args.output_dir + args.task_name.lower() + args.output_name, map_location=device)

        # Make Lilbert.
        if args.method == 'prune':  # 2
            student, pruner = lilbert.make_lil_bert(teacher, fraction=args.prune_frac, fraction_emb=args.prune_frac_emb,
                                                    method="prune",
                                                    vanilla=args.from_scratch), None
        elif args.method == 'cut':  # 3
            student, pruner = lilbert.make_lil_bert(teacher, dim=args.student_dim, method="cut",
                                                    vanilla=args.from_scratch), None
        elif args.method == 'linear_pruning':
            student, pruner = lilbert.LinearPruner.init(teacher, t_total=args.t_total, n_steps=args.n_steps)
        else:
            raise gd.UnknownMethod(f"Method {args.method} is not known.")
        args.pruner = pruner

        # Loss function
        if args.loss_type == 'attention':
            model = lilbert.BertDistillWithAttentionModel(teacher, student, alpha=args.alpha)
        elif args.loss_type == 'distill':
            model = lilbert.BertDistillModel(teacher, student, alpha=args.alpha)
        else:
            raise gd.UnknownLossType(f"The loss method {args.loss_type} is not known.")

    else:
        raise gd.UnknownMode(f"{args.mode} is not a known mode.")

    # Setting up wandb
    if args.call_wandb:
        wandb.init(project="lilbert",
                   notes=args.get('notes', ''))
        for k, v in args.items():
            wandb.config[k] = v

    main(args=args, model=model, tokenizer=tok)
