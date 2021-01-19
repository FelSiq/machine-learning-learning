# Following: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import torch
import torch.nn as nn
import tqdm

import utils


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_in_vocab: int,
        n_out_vocab: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        device: str,
    ):
        super(TransformerModel, self).__init__()

        self.model_type = "Transformer"

        self.input_X = nn.Sequential(
            nn.Embedding(n_in_vocab, d_model),
            utils.PositionalEncoding(d_model, max_len=d_model),
        )

        self.input_Y = nn.Sequential(
            nn.Embedding(n_in_vocab, d_model),
            utils.PositionalEncoding(d_model, max_len=11),
        )

        self.transformer = nn.Transformer(
            d_model, nhead, num_layers, num_layers, dim_feedforward
        )

        self.linear = nn.Linear(d_model, n_out_vocab)

        self._mask = torch.triu(
            torch.full((11, 11), float("-inf"), device=device), diagonal=1
        )

    def forward(self, X, Y):
        X = self.input_X(X)
        Y = self.input_Y(Y)
        X = self.transformer(X, Y, tgt_mask=self._mask)
        X = self.linear(X)
        return X


def predict(model, sentence, device, human, machine, inv_machine):
    model.eval()
    prepared, _ = utils.prepare_data([(sentence, "")], human, machine)
    print("Test input:", sentence)
    print(prepared)
    prepared = prepared.to(device)
    out = torch.zeros((1, 11), device=device, dtype=torch.long)
    out[0][0] = human["<pad>"]

    out = torch.transpose(out, 0, 1)
    prepared = torch.transpose(prepared, 0, 1)

    for i in range(1, 1 + 10):
        pred = model(prepared, out)
        ind = pred[i - 1].squeeze().argmax(dim=-1)
        out[i][0] = ind.item()

    out = out.squeeze().detach().cpu().numpy()

    print("Output (raw):", out)
    print("Output:", "".join([inv_machine[i.item()] for i in out[1:]]))


def _test():
    data_size = 8000
    eval_size = 1000
    train_epochs = 5
    device = "cuda"
    initial_epoch = 0
    max_size = 80
    dim_feedforward = 256
    nhead = 8
    insert_noise = False

    train_X, human, machine, inv_machine = utils.load_dataset(data_size, insert_noise)
    print("Train samples:")
    print(train_X[:3])
    train_X, train_Y = utils.prepare_data(train_X, human, machine, max_size)

    eval_X = utils.load_dataset(eval_size, insert_noise)[0]
    print("Eval samples:")
    print(eval_X[:3])
    eval_X, eval_Y = utils.prepare_data(eval_X, human, machine, max_size)

    model = TransformerModel(
        n_in_vocab=len(human),
        n_out_vocab=len(machine),
        d_model=max_size,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=2,
        device=device,
    )

    model = model.to(device)
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    eval_X = eval_X.to(device)
    eval_Y = eval_Y.to(device)

    optim = torch.optim.Adam(model.parameters(), 0.001)

    torch_train_X = torch.utils.data.TensorDataset(train_X, train_Y)
    train_dataloader = torch.utils.data.DataLoader(
        torch_train_X, batch_size=128, shuffle=True
    )

    torch_eval_X = torch.utils.data.TensorDataset(eval_X, eval_Y)
    eval_dataloader = torch.utils.data.DataLoader(
        torch_eval_X, batch_size=128, shuffle=False
    )

    criterion = nn.CrossEntropyLoss(ignore_index=human["<pad>"])

    for epoch in range(initial_epoch + 1, initial_epoch + 1 + train_epochs):
        print(f"Epoch: {epoch:4d} / {initial_epoch + train_epochs} ...")

        train_batch_loss = train_acc = 0.0
        total_batches = 0
        model.train()

        for X_batch, Y_batch in tqdm.auto.tqdm(train_dataloader):
            optim.zero_grad()

            X_batch = torch.transpose(X_batch, 0, 1)
            Y_batch = torch.transpose(Y_batch, 0, 1)

            y_preds = model(X_batch, Y_batch)

            y_preds = y_preds[:-1].view(-1, len(machine))
            Y_batch = Y_batch[1:].reshape(-1)

            loss = criterion(y_preds, Y_batch)
            loss.backward()
            optim.step()

            train_batch_loss += loss.item()
            train_acc += (y_preds.argmax(dim=-1) == Y_batch).float().mean().item()
            total_batches += 1

        train_batch_loss /= total_batches
        train_acc /= total_batches

        eval_batch_loss = eval_acc = 0.0
        total_batches = 0
        model.eval()

        for X_batch, Y_batch in tqdm.auto.tqdm(eval_dataloader):
            X_batch = torch.transpose(X_batch, 0, 1)
            Y_batch = torch.transpose(Y_batch, 0, 1)

            y_preds = model(X_batch, Y_batch)

            y_preds = y_preds[:-1].view(-1, len(machine))
            Y_batch = Y_batch[1:].reshape(-1)

            loss = criterion(y_preds, Y_batch)

            eval_batch_loss += loss.item()
            eval_acc += (y_preds.argmax(dim=-1) == Y_batch).float().mean().item()
            total_batches += 1

        eval_batch_loss /= total_batches
        eval_acc /= total_batches

        print(f"train loss : {train_batch_loss:4.4f} - train acc: {train_acc:4.4f}")
        print(f"eval loss : {eval_batch_loss:4.4f} - eval acc: {eval_acc:4.4f}")

    test_input = "Tuesday, January 19, 2021"
    predict(model, test_input, device, human, machine, inv_machine)


if __name__ == "__main__":
    _test()
