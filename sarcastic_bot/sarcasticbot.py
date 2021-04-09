import discord
import random
import transformers
from utils import get_tokenizer
import numpy as np
import torch as pt

client = discord.Client()

model = None
tokenizer = None


def get_reply(comment: str) -> str:
    global model
    global tokenizer
    # encode and truncate comment if it's bigger than the window size of gpt2
    comment = tokenizer.encode(
        comment,
        return_tensors='pt',
        truncation=True,
        max_length=1024)

    # use GPU, if available
    if pt.cuda.is_available():
        model.to(pt.device('cuda:0'))
        comment = comment.to(pt.device('cuda:0'))

    # max length of output = 100 words (the remaining length is the parent
    # comment itself by the design of training)
    max_length = len(comment) + 200

    output = model.generate(
        comment,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True
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


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))
    # load model and tokenizer
    global model
    global tokenizer
    try:
        model = pt.load("./model_1.pt", map_location=pt.device('cpu'))
        model.eval()
        tokenizer = get_tokenizer()
    except BaseException:
        print("Service unavailable.")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if (client.user in message.mentions):
        if (message.reference is not None):
            referedmsg = await message.channel.fetch_message(message.reference.message_id)
            reply = get_reply(referedmsg.content)
            await referedmsg.reply(reply)
        else:
            await message.channel.send("Hi, there.")

# Please insert your private API key and make sure that it's not visible
# to the public.
client.run( < INSERT API KEY > )
