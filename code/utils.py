import json
import os
import re
import string
import sys
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from models import Prober

logger = logging.getLogger(__name__)
MAX_NUM_VECTORS = 10


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


class QAData(object):
    def __init__(self, logger, args, data_path, is_training, template):
        self.data_path = data_path
        if args.debug:
            self.data_path = data_path.replace("train", "trial")
        if "/test" in self.data_path:
            self.data_type = "test"
        elif "/dev" in self.data_path:
            self.data_type = "dev"
        elif "/train" in self.data_path:
            self.data_type = "train"
        else:
            self.data_type = "trial"
        self.data = []
        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args
        self.template = template
        self.metric = "EM"
        self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        with open(self.data_path, 'r') as f:
            raw_samples = json.load(f)
        for data_sample in raw_samples['data']:
            for paragraph in data_sample['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    q = qa['question']
                    a = [answer['text'] for answer in qa['answers']]
                    feature_sample = self.gen_feature_sample(context, q, a)
                    self.data.append(feature_sample)

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        text = self.tokenizer.decode(tokens,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip()
        return ''.join(text.split())


    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def parse_template(self, feature_sample):
        Question_SYMBOL = "[Q]"
        Context_SYMBOL = "[C]"
        template = self.template.replace(Question_SYMBOL, feature_sample['question'])
        template = template.replace(Context_SYMBOL, feature_sample['context'])
        return template

    def gen_feature_sample(self, context, question, answer):
        feature_sample = {}
        feature_sample['context'] = context
        feature_sample['question'] = question
        feature_sample['answer'] = answer
        feature_sample['input_sentences'] = self.parse_template(feature_sample)
        return feature_sample

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            assert type(answer) == list
            metadata.append((len(new_answers), len(new_answers) + len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(
                ".tsv" if self.data_path.endswith(".tsv") else ".json",
                "{}-{}-{}.json".format(
                    "-uncased" if self.args.do_lowercase else "",
                    self.args.relation,
                    postfix)))
        if self.load and os.path.exists(preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                metadata = json.load(f)
        else:
            logger.info("Start tokenizing...")
            questions = [d["input_sentences"] for d in self.data]
            answers = [d["answer"] for d in self.data]
            answers, metadata = self.flatten(answers)
            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]
            if self.args.append_another_bos:
                questions = ["<s> " + question for question in questions]
                answers = ["<s> " + answer for answer in answers]
            question_input = tokenizer.batch_encode_plus(questions,
                                                         padding='max_length',
                                                         truncation=True,
                                                         max_length=self.args.max_input_length)
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       padding='max_length',
                                                       truncation=True,
                                                       max_length=self.args.max_output_length)
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            if self.load:
                preprocessed_data = [input_ids, attention_mask,
                                     decoder_input_ids, decoder_attention_mask, metadata]
                with open(preprocessed_path, "w") as f:
                    json.dump(preprocessed_data, f)

        self.dataset = MyQADataset(input_ids, attention_mask,
                                   decoder_input_ids, decoder_attention_mask,
                                   in_metadata=None, out_metadata=metadata,
                                   is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions):
        assert len(predictions) == len(self), (len(predictions), len(self))
        ems = []
        for (prediction, dp) in zip(predictions, self.data):
            ems.append(get_exact_match(prediction, dp["answer"]))
        return ems

    def save_predictions(self, predictions):
        assert len(predictions) == len(self), (len(predictions), len(self))
        # prediction_dict = {dp["id"]:prediction for dp, prediction in zip(self.data, predictions)}
        save_path = os.path.join(self.args.output_dir, "predictions.json")
        with open(save_path, "w") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
        self.logger.info("Saved prediction in {}".format(save_path))


def get_exact_match(prediction, groundtruth):
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return normalize_answer(prediction) == normalize_answer(groundtruth)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


class MyQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1 + len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1 + len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids) == len(self.attention_mask) == self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids) == len(self.decoder_attention_mask) == self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
               self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]


class MyDataLoader(DataLoader):
    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.eval_batch_size
        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)


def convert_tokens_to_string(tokens):
    out_string = " ".join(tokens).replace(" ##", "").strip()
    return out_string


def get_relation_meta(args):
    relations = load_file(args.relation_profile)
    for relation in relations:
        if relation['relation'] == args.relation:
            return relation
    raise ValueError('Relation info %s not found in file %s' % (args.relation, args.relation_profile))


def get_new_token(vid):
    assert (vid > 0 and vid <= MAX_NUM_VECTORS)
    return '[V%d]' % (vid)


def convert_manual_to_dense(manual_template, model):
    def assign_embedding(new_token, token):
        """
        assign the embedding of token to new_token
        """
        logger.info('Tie embeddings of tokens: (%s, %s)' % (new_token, token))
        id_a = model.tokenizer.convert_tokens_to_ids([new_token])[0]
        id_b = model.tokenizer.convert_tokens_to_ids([token])[0]
        with torch.no_grad():
            model.base_model.shared.weight[id_a] = model.base_model.shared.weight[id_b].detach().clone()

    new_token_id = 0
    template = []
    for word in manual_template.split():
        if word in ['[Q]', '[A]', '[C]']:
            template.append(word)
        else:
            tokens = model.tokenizer.tokenize(' ' + word)
            for token in tokens:
                new_token_id += 1
                template.append(get_new_token(new_token_id))
                assign_embedding(get_new_token(new_token_id), token)

    return ' '.join(template)


def init_template(args, model):
    if args.init_manual_template:
        relation = get_relation_meta(args)
        template = convert_manual_to_dense(relation['template'], model)
    else:
        template = '[Q] ' + ' '.join(['[V%d]' % (i + 1) for i in range(args.num_vectors)]) + ' [C] .'
    return template


def prepare_for_dense_prompt(model):
    new_tokens = [get_new_token(i + 1) for i in range(MAX_NUM_VECTORS)]
    model.tokenizer.add_tokens(new_tokens)
    ebd = model.model.resize_token_embeddings(len(model.tokenizer))
    model.model.config.vocab_size = len(model.tokenizer)
    logger.info('# vocab after adding new tokens: %d' % len(model.tokenizer))


def load_optiprompt(args):
    # load bert model (pre-trained)
    model = Prober(args, random_init=args.random_init)
    original_vocab_size = len(list(model.tokenizer.get_vocab()))
    prepare_for_dense_prompt(model)

    if os.path.exists(os.path.join(args.output_dir, 'prompt_vecs.npy')) and not os.path.exists(os.path.join(args.output_dir, 'best-model.pth')):
        logger.info("Loading OptiPrompt's [V]s")
        with open(os.path.join(args.output_dir, 'prompt_vecs.npy'), 'rb') as f:
            vs = np.load(f)

        # copy fine-tuned new_tokens to the pre-trained model
        with torch.no_grad():
            model.base_model.shared.weight[original_vocab_size:] = torch.Tensor(vs)

    elif os.path.exists(os.path.join(args.output_dir, 'best-model.pth')):
        logger.info("Loading model from checkpoint")
        model.model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best-model.pth')))

    return model


def save_optiprompt(args, model):
    logger.info("Saving OptiPrompt's [V]s..")
    vs = model.base_model.shared.weight[args.original_vocab_size:].detach().cpu().numpy()
    with open(os.path.join(args.output_dir, 'prompt_vecs.npy'), 'wb') as f:
        np.save(f, vs)


def save_model(Model, args):
    logger.info('Saving model...')
    model_to_save = Model.model
    model_to_save.save_pretrained(args.output_dir)
    Model.tokenizer.save_pretrained(args.output_dir)
