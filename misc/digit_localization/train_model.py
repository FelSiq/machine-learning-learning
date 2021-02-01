import os
import functools

import torch.nn as nn
import torch
import tqdm

import config
import utils


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.weights = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, config.TARGET_DEPTH, kernel_size=1, stride=1),
        )

    def forward(self, X):
        # Output shape: (?, 15, 20, 20)
        return self.weights(X)

    @staticmethod
    def split_output(out):
        is_object = out[:, 0, ...]
        coords = out[:, [1, 2], ...]
        dims = out[:, [3, 4], ...]
        class_logits = out[:, 5:, ...]
        return is_object, coords, dims, class_logits


def get_data():
    X_train = torch.load(os.path.join(config.DATA_DIR, "insts_0.pt"))
    y_train = torch.load(os.path.join(config.DATA_DIR, "targets_0.pt"))

    X_train = X_train.unsqueeze(1)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

    return train_dataset


def loss_func(y_preds, y_true, is_object_weight: float = 1.0):
    true_is_object, true_coords, true_dims, true_class_logits = Model.split_output(
        y_true
    )
    preds_is_object, preds_coords, preds_dims, preds_class_logits = Model.split_output(
        y_preds
    )

    # Note: no need to shrink 'preds_is_object' because loss function
    # (bce loss w/ logits) already incorporate the sigmoid function.
    total_loss = is_object_weight * nn.functional.binary_cross_entropy_with_logits(
        preds_is_object, true_is_object
    )

    preds_coords = torch.exp(preds_coords)  # Note: ensure > 0
    preds_dims = torch.exp(preds_dims)  # Note: ensure > 0

    true_class_logits = true_class_logits.argmax(dim=1, keepdims=True)

    # Note: when the true label of a cell does not has any object within it, then
    # the remaining predicted values does not matter as long as the model
    # detected no object in the first place. Therefore, here we calculate the
    # remaining losses only for cells that actually has some object (based on
    # the ground truth).
    is_object_mask = (true_is_object >= 0.999).unsqueeze(1)
    preds_coords = torch.masked_select(preds_coords, is_object_mask)
    preds_dims = torch.masked_select(preds_dims, is_object_mask)
    preds_class_logits = torch.masked_select(preds_class_logits, is_object_mask)
    true_coords = torch.masked_select(true_coords, is_object_mask)
    true_dims = torch.masked_select(true_dims, is_object_mask)
    true_class_logits = torch.masked_select(true_class_logits, is_object_mask)

    total_loss += nn.functional.binary_cross_entropy_with_logits(
        preds_coords, true_coords
    )
    total_loss += nn.functional.mse_loss(preds_dims, true_dims)

    preds_class_logits = preds_class_logits.view(-1, config.NUM_CLASSES)

    # Note: no need to softmax class logits since the loss functiona
    # (cross entropy) already incorporate the (log-)softmax function.
    total_loss += nn.functional.cross_entropy(preds_class_logits, true_class_logits)

    return total_loss


def predict(model, X):
    model.eval()
    y_preds = model(X)
    is_object, coords, dims, class_logits = model.split_output(y_preds)

    y_preds[:, 0, ...] = torch.sigmoid(is_object)
    y_preds[:, [1, 2], ...] = torch.sigmoid(coords)
    y_preds[:, [3, 4], ...] = torch.exp(dims)
    y_preds[:, 5:, ...] = torch.softmax(class_logits, dim=1)

    return y_preds


def _test():
    train_epochs = 250
    epochs_per_checkpoint = 20
    device = "cuda"
    train_batch_size = 32
    checkpoint_path = "dl_checkpoint.tar"

    train_dataset = get_data()

    model = Model()
    optim = torch.optim.Adam(model.parameters(), 4e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, factor=0.95, patience=15, verbose=True
    )

    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded checkpoint.")

    except FileNotFoundError:
        pass

    criterion = functools.partial(loss_func, is_object_weight=3.0)
    model = model.to(device)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=train_batch_size
    )

    for epoch in range(1, 1 + train_epochs):
        print(f"Epoch: {epoch:4d} / {train_epochs}...")
        model.train()
        train_total_batches = 0
        train_loss = 0.0

        for X_batch, y_batch in tqdm.auto.tqdm(train_dataloader):
            optim.zero_grad()

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_preds = model(X_batch)
            loss = criterion(y_preds, y_batch)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optim.step()

            with torch.no_grad():
                train_total_batches += 1
                train_loss += loss.item()

        train_loss /= train_total_batches
        print(f"train loss: {train_loss:.4f}")
        scheduler.step(train_loss)

        if (
            epochs_per_checkpoint > 0 and epoch % epochs_per_checkpoint == 0
        ) or epoch == train_epochs:
            checkpoint = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            print("Saved checkpoint.")

    insts_eval = X_batch[:16]
    y_preds = predict(model, insts_eval)

    for inst, pred in zip(insts_eval, y_preds):
        print(pred[0, ...].max())
        utils.plot_instance(inst.detach().cpu().squeeze(), pred.detach().cpu())

    print("Done.")


if __name__ == "__main__":
    _test()
