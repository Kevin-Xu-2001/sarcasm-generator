import torch
from torchtext.data.metrics import bleu_score
import transformers
from transformers import GPT2Tokenizer
import random
import numpy as np


def get_tokenizer():
    """ Returns GPT2 tokenizer after adding separator and padding tokens """

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {
        'pad_token': '<|pad|>',
        'sep_token': '<|sep|>',
        'eos_token': '<|eos|>'}
    num_add_toks = tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def gen_reply(model, tokenizer, comment, method="beam_search"):
    # truncate comment if it's bigger than the window size of gpt2
    if len(comment) > 1024:
        comment = comment[:1024]

    # max length of output = 100 words (the remaining length is the parent
    # comment itself by the design of training)
    max_length = len(comment) + 200

    if method == "beam_search":
        output = model.generate(
            comment,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            num_return_sequences=5,
            early_stopping=True
        )
    elif method == "top_k_sampling":
        output = model.generate(
            comment,
            do_sample=True,
            max_length=max_length,
            top_k=50
        )
    elif method == "top_p_sampling":
        output = model.generate(
            comment,
            do_sample=True,
            max_length=max_length,
            top_p=0.9,
            top_k=0
        )

    else:  # greedy
        output = model.generate(
            comment,
            max_length=max_length
        )

    output = tokenizer.decode(output[0])

    output = output.lower()
    output = output.split('<|eos|>')

    # ignore the first output because it's same as the parent comment
    # also ignore the last output since it's incomplete (no <|eos|> at the end)
    output = output[1:-1]

    # strip white spaces
    output = list(map(lambda x: x.strip(), output))

    # return a random reply out of those generated
    output = output[np.random.randint(0, len(output))]
    return output


def eval_bleu(model, tokenizer, comments, replies):
    gen_replies = []
    for comment in comments:
        comment_ = comment.lower()
        inp = tokenizer.encode(comment_, return_tensors='pt')
        reply = gen_reply(model, tokenizer, inp)

        reply = reply.split()
        reply = list(map(lambda x: x.strip(',.;:?!\'\" '), reply))

        gen_replies.append(reply)

    ref_replies = []
    for reply in replies:
        reply_ = reply.lower()
        reply_ = reply_.split()
        reply_ = list(map(lambda x: x.strip(',.;:?!\'\" '), reply_))
        ref_replies.append([reply_])

    return bleu_score(gen_replies, ref_replies)
