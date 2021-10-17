import os
import json
import logging
import random

from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertConfig
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
from transformers import BartForConditionalGeneration, BartConfig
from transformers import AutoConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np

logger = logging.getLogger(__name__)


class MyBart(BartForConditionalGeneration):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # Added for compatibility with 4.4.2
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:  # TJH added for compatibility with other 4.4.2 seq2seq models
            if decoder_input_ids is None:
                decoder_start_token_id = self.config.decoder_start_token_id
                decoder_input_ids = labels.new_zeros(labels.shape)
                decoder_input_ids[..., 1:] = labels[..., :-1].clone()
                decoder_input_ids[..., 0] = decoder_start_token_id

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduce=False)
            losses = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              labels.view(-1))
            loss = torch.sum(losses * decoder_attention_mask.float().view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate_from_string(self, _input, tokenizer=None, **generator_args):
        assert tokenizer is not None
        if isinstance(_input, str):
            _input = [[0] + tokenizer.encode(_input)]
        if isinstance(_input, list) and isinstance(_input[0], str):
            _input = [[0] + tokenizer.encode(i) for i in _input]
        if isinstance(_input, list):
            if isinstance(_input[0], int):
                _input = [_input]
            _input = torch.LongTensor(_input)
        res = self.generate(_input, **generator_args)
        return ([tokenizer.decode(x, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True).strip() for x in res])


class Prober():

    def __init__(self, args, random_init='none'):
        assert(random_init in ['none', 'all', 'embedding'])

        super().__init__()

        self._model_device = 'cpu'

        model_name = args.model_name
        vocab_name = model_name

        if args.model_dir is not None:
            # load bert model from file
            model_name = str(args.model_dir) + "/"
            vocab_name = model_name
            logger.info("loading BERT model from {}".format(model_name))

        # Load pre-trained model tokenizer (vocabulary)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(args.seed)

        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, AlbertConfig):
            self.model_type = 'albert'
            self.tokenizer = AlbertTokenizer.from_pretrained(vocab_name)
            self.mlm_model = AlbertForMaskedLM.from_pretrained(model_name)
            if random_init == 'all':
                logger.info('Random initialize model...')
                self.mlm_model = AlbertForMaskedLM(self.mlm_model.config)
            self.base_model = self.mlm_model.albert
        elif isinstance(config, RobertaConfig):
            self.model_type = 'roberta'
            self.tokenizer = RobertaTokenizer.from_pretrained(vocab_name)
            self.mlm_model = RobertaForMaskedLM.from_pretrained(model_name)
            if random_init == 'all':
                logger.info('Random initialize model...')
                self.mlm_model = RobertaForMaskedLM(self.mlm_model.config)
            self.base_model = self.mlm_model.roberta
        elif isinstance(config, BertConfig):
            self.model_type = 'bert'
            self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
            self.mlm_model = BertForMaskedLM.from_pretrained(model_name)
            if random_init == 'all':
                logger.info('Random initialize model...')
                self.mlm_model = BertForMaskedLM(self.mlm_model.config)
            self.base_model = self.mlm_model.bert
        elif isinstance(config, BartConfig):
            self.model_type = 'bart'
            self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
            self.model = MyBart.from_pretrained(model_name)
            self.base_model = self.model.model
        else:
            raise ValueError('Model %s not supported yet!'%(model_name))

        self.model.eval()

        # original vocab
        self.map_indices = None
        self.vocab = list(self.tokenizer.get_vocab().keys())
        logger.info('Vocab size: %d'%len(self.vocab))
        self._init_inverse_vocab()

        self.MASK = self.tokenizer.mask_token
        self.EOS = self.tokenizer.eos_token
        self.CLS = self.tokenizer.cls_token
        self.SEP = self.tokenizer.sep_token
        self.UNK = self.tokenizer.unk_token
        # print(self.MASK, self.EOS, self.CLS, self.SEP, self.UNK)

        self.pad_id = self.inverse_vocab[self.tokenizer.pad_token]
        self.unk_index = self.inverse_vocab[self.tokenizer.unk_token]

    def _cuda(self):
        self.model.cuda()

    def try_cuda(self):
        """Move model to GPU if one is available."""
        if torch.cuda.is_available():
            if self._model_device != 'cuda':
                logger.info('Moving model to CUDA')
                self._cuda()
                self._model_device = 'cuda'
        else:
            logger.info('No CUDA found')

    def _init_inverse_vocab(self):
        self.inverse_vocab = {w: i for i, w in enumerate(self.vocab)}
