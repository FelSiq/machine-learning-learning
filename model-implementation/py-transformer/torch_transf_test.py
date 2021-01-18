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
    ):
        super(TransformerModel, self).__init__()

        self.model_type = "Transformer"

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)

        self.weights = nn.Sequential(
            nn.Embedding(n_in_vocab, d_model),
            utils.PositionalEncoding(d_model, max_len=d_model),
            nn.TransformerEncoder(encoder_layers, num_layers),
            nn.Linear(d_model, n_out_vocab),
        )

    def forward(self, X):
        return self.weights(X)


def _test():
    data_size = 8000
    eval_size = 1000
    train_epochs = 2
    device = "cuda"
    initial_epoch = 0
    max_size = 128

    train_dataset, human, machine, inv_machine = utils.load_dataset(data_size)
    train_dataset, train_masks = utils.prepare_data(
        train_dataset, human, machine, max_size
    )

    eval_dataset = utils.load_dataset(eval_size)[0]
    eval_dataset, eval_masks = utils.prepare_data(
        eval_dataset, human, machine, max_size
    )

    model = TransformerModel(
        n_in_vocab=len(human),
        n_out_vocab=len(machine),
        d_model=max_size,
        nhead=2,
        dim_feedforward=16,
        num_layers=2,
    )

    model = model.to(device)
    train_dataset = train_dataset.to(device)
    train_masks = train_masks.to(device)
    eval_dataset = eval_dataset.to(device)
    eval_masks = eval_masks.to(device)

    optim = torch.optim.Adam(model.parameters(), 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        step_size=1.0,
        gamma=0.95,
    )

    torch_train_dataset = torch.utils.data.TensorDataset(train_dataset, train_masks)
    train_dataloader = torch.utils.data.DataLoader(
        torch_train_dataset, batch_size=128, shuffle=True
    )

    torch_eval_dataset = torch.utils.data.TensorDataset(eval_dataset, eval_masks)
    eval_dataloader = torch.utils.data.DataLoader(
        torch_eval_dataset, batch_size=128, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(initial_epoch + 1, initial_epoch + 1 + train_epochs):
        print(f"Epoch: {epoch:4d} / {initial_epoch + train_epochs} ...")

        batch_loss = 0.0
        total_epochs = 0
        model.train()

        for batch, mask in tqdm.auto.tqdm(train_dataloader):
            optim.zero_grad()

            batch = torch.transpose(batch, 0, 1)
            mask = torch.transpose(mask, 0, 1)

            y_preds = model(batch)
            mask_expanded = mask.expand(len(machine), *mask.shape).permute(1, 2, 0)
            y_preds = torch.masked_select(y_preds, mask_expanded)
            y_preds = y_preds.reshape(-1, len(machine))

            batch_masked = torch.masked_select(batch, mask)
            loss = criterion(y_preds, batch_masked)

            loss.backward()
            optim.step()

            batch_loss += loss.item()
            total_epochs += 1

        batch_loss /= total_epochs

        eval_loss = 0.0
        total_epochs = 0
        eval_acc = 0.0
        model.eval()

        for batch, mask in tqdm.auto.tqdm(eval_dataloader):
            batch = torch.transpose(batch, 0, 1)
            mask = torch.transpose(mask, 0, 1)

            y_preds = model(batch)
            mask_expanded = mask.expand(len(machine), *mask.shape).permute(1, 2, 0)
            y_preds = torch.masked_select(y_preds, mask_expanded)
            y_preds = y_preds.reshape(-1, len(machine))

            batch_masked = torch.masked_select(batch, mask)
            loss = criterion(y_preds, batch_masked)

            eval_loss += loss.item()
            total_epochs += 1

            preds = y_preds.argmax(dim=-1).squeeze()
            eval_acc += (preds == batch_masked).float().mean().item()

        eval_loss /= total_epochs
        eval_acc /= total_epochs

        scheduler.step(eval_loss)

        print(f"train loss : {batch_loss:04.4f}")
        print(f"eval loss  : {eval_loss:04.4f} - eval acc : {eval_acc:04.4f}")


if __name__ == "__main__":
    _test()
