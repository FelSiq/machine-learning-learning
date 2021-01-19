# Sources:
# 1. https://www.coursera.org/learn/nlp-sequence-models/home/week/3
# 2. https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import math
import random

import numpy as np
from faker import Faker
from tqdm import tqdm
from babel.dates import format_date
import torch
import torch.nn as nn

fake = Faker()
# Faker.seed(12345)
# random.seed(12345)

# Define format of the data we would like to generate
FORMATS = [
    "short",
    "medium",
    "long",
    "full",
    "full",
    "full",
    "full",
    "full",
    "full",
    "full",
    "full",
    "full",
    "full",
    "d MMM YYY",
    "d MMMM YYY",
    "dd MMM YYY",
    "d MMM, YYY",
    "d MMMM, YYY",
    "dd, MMM YYY",
    "d MM YY",
    "d MMMM YYY",
    "MMMM d YYY",
    "MMMM d, YYY",
    "dd.MM.YY",
]

# change this if you want it to work with another language
LOCALES = ["en_US"]


class PositionalEncoding(nn.Module):
    # Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.0, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def insert_noise(human_readable):
    random_start_num = random.randint(0, 5)
    random_end_num = random.randint(0, 5)

    human_readable = list(human_readable)

    noise_ind = np.random.random(size=len(human_readable)) <= 0.05
    noise_ind = np.flatnonzero(noise_ind)

    for i in noise_ind:
        human_readable[i] = chr(random.randind(0, 256))

    for _ in range(random_start_num):
        noise = [chr(random.randint(0, 256)) for _ in range(random.randint(1, 10))]
        noise.append(" ")
        noise.extend(human_readable)
        human_readable = noise

    for _ in range(random_end_num):
        noise = [chr(random.randint(0, 256)) for _ in range(random.randint(1, 10))]
        noise.insert(0, " ")
        human_readable.extend(noise)

    return "".join(human_readable)


def load_date(insert_noise: bool):
    """
    Loads some fake dates
    :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(
            dt, format=random.choice(FORMATS), locale="en_US"
        )  # locale=random.choice(LOCALES))
        if insert_noise:
            human_readable = insert_noise(human_readable)
        machine_readable = dt.isoformat()

    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt


def load_dataset(m, insert_noise: bool = False):
    """
    Loads a dataset with m examples and vocabularies
    :m: the number of examples to generate
    """

    human_vocab = set()
    machine_vocab = set()
    dataset = []

    for i in tqdm(range(m)):
        h, m, _ = load_date(insert_noise)
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))

    human = dict(
        zip(
            sorted(human_vocab) + ["<unk>", "<pad>", "<eos>"],
            list(range(len(human_vocab) + 3)),
        )
    )
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v: k for k, v in inv_machine.items()}

    return dataset, human, machine, inv_machine


def prepare_data(dataset, vocab_x, vocab_y, max_size: int = 80):
    X_data = []
    Y_data = []

    eos_tok = vocab_x["<eos>"]
    unk_tok_x = vocab_x["<unk>"]
    pad_id = vocab_x["<pad>"]

    for X, y in dataset:
        x = [vocab_x.get(c, unk_tok_x) for c in X]
        x.append(eos_tok)
        x += [pad_id] * (max_size - len(x))
        y = [pad_id] + [vocab_y[c] for c in y]

        if len(x) <= max_size:
            X_data.append(x)
            Y_data.append(y)

    X_data = torch.tensor(X_data, dtype=torch.long)
    Y_data = torch.tensor(Y_data, dtype=torch.long)

    return X_data, Y_data
