import os
import numpy as np
from tqdm import tqdm

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from models import Prober
from utils import *


def run(args, logger):
    logger.info('***Build model: %s***' % args.model_name)

    Model = Prober(args, random_init=args.random_init)
    original_vocab_size = Model.tokenizer.vocab_size
    args.original_vocab_size = original_vocab_size
    logger.info('Original vocab size: %d' % original_vocab_size)
    prepare_for_dense_prompt(Model)

    if args.n_gpu > 1:
        Model.model = torch.nn.DataParallel(Model.model)
    if args.n_gpu > 0:
        Model.model.to(torch.device("cuda"))

    template = init_template(args, Model)
    logger.info('Template: %s' % template)

    if os.path.exists(os.path.join(args.output_dir, 'best-model.pth')):
        logger.info("Loading model from checkpoint")
        Model.model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best-model.pth')))

    elif os.path.exists(os.path.join(args.output_dir, 'prompt_vecs.npy')):
        logger.info("Loading OptiPrompt's [V]s")
        with open(os.path.join(args.output_dir, 'prompt_vecs.npy'), 'rb') as f:
            vs = np.load(f)

        # copy fine-tuned new_tokens to the pre-trained model
        with torch.no_grad():
            Model.base_model.shared.weight[original_vocab_size:] = torch.Tensor(vs)

    if args.do_train:
        # Prepare train/valid data
        train_data = QAData(logger, args, args.train_data, True, template)
        dev_data = QAData(logger, args, args.dev_data, False, template)
        train_data.load_dataset(Model.tokenizer)
        train_data.load_dataloader()
        dev_data.load_dataset(Model.tokenizer)
        dev_data.load_dataloader()

        if args.freeze:
            # Add word embeddings to the optimizer
            logger.info('Freeze model parameter')
            optimizer = AdamW([{'params': Model.base_model.shared.parameters()}], lr=args.learning_rate,
                              correct_bias=False)
        else:
            logger.info('Update model parameter')
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in Model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in Model.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        t_total = len(train_data.dataloader) * args.num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * args.warmup_proportion),
                                                    num_training_steps=t_total)

        train(args, logger, Model.model, train_data, dev_data, optimizer, scheduler)

    if args.do_eval:
        logger.info('***Evaluate***')
        Model = load_optiprompt(args)
        dev_data = QAData(logger, args, args.dev_data, False, template)
        dev_data.load_dataset(Model.tokenizer)
        dev_data.load_dataloader()
        ems = inference(Model.model, dev_data, save_predictions=True)
        logger.info("%s on %s data: %.2f" % (dev_data.metric, dev_data.data_type, ems * 100))


def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    best_accuracy = -1
    logger.info("***Valid set before train***")
    best_accuracy = inference(model, dev_data)
    logger.info("%s on %s data: %.2f" % (dev_data.metric, dev_data.data_type, best_accuracy * 100))

    # save_optiprompt(args, model)

    logger.info("***Starting training***")
    global_step = 0
    wait_step = 0
    train_losses = []
    eval_step = len(train_data.dataloader) // args.eval_per_epoch
    stop_training = False
    model.train()
    for epoch in range(args.num_epoch):
        for i, batch in tqdm(enumerate(train_data.dataloader)):
            batch = [b.to(torch.device("cuda")) for b in batch]
            outputs = model(input_ids=batch[0], attention_mask=batch[1],
                            labels=batch[2], decoder_attention_mask=batch[3])
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # if args.n_gpu > 1:
            #     loss = loss.mean()
            train_losses.append(loss.detach().cpu())
            loss.backward()

            global_step += 1

            # set normal tokens' gradients to be zero
            for p in model.base_model.shared.parameters():
                # only update new tokens
                p.grad[:args.original_vocab_size, :] = 0.0

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if args.check_step > 0 and ((global_step + 1) % args.check_step == 0):
                logger.info('Epoch=%d, iter=%d, loss=%.5f' % (epoch, i, np.mean(train_losses)))
                sys.stdout.flush()
                train_losses = []

            if eval_step > 0 and (global_step + 1) % eval_step == 0:
                # Eval during training
                curr_em = inference(model if args.n_gpu == 1 else model.module, dev_data)
                logger.info("Step %d Train loss %.2f %s %.2f%% on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    dev_data.metric,
                    curr_em * 100,
                    epoch))
                train_losses = []
                if best_accuracy < curr_em:
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                                (dev_data.metric, best_accuracy * 100.0, curr_em * 100.0, epoch, global_step))
                    if args.freeze:
                        save_optiprompt(args, model)
                    else:
                        model_state_dict = {k: v.cpu() for (k, v) in model.state_dict().items()}
                        # if args.n_gpu > 1:
                        #     model_state_dict = convert_to_single_gpu(model_state_dict)
                        torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pth"))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break


def inference(model, dev_data, save_predictions=False):
    model.eval()
    predictions = []
    for i, batch in enumerate(dev_data.dataloader):
        batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 min_length=1,
                                 max_length=dev_data.args.max_output_length,
                                 early_stopping=True, )
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return np.mean(dev_data.evaluate(predictions))