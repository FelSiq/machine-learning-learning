import functools

import torch.nn as nn
import torch
import tqdm

import config
import utils


class Model(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super(Model, self).__init__()

        self.weights = nn.Sequential(
            self._create_block(1, 64, 5, 2, dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._create_block(64, 256, 5, 2, dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._create_block(256, 128, 5, 2, dropout),
            self._create_block(128, 1024, 1, 1, dropout),
            nn.Conv2d(1024, config.TARGET_DEPTH, kernel_size=1, stride=1),
        )

    def forward(self, X):
        return self.weights(X)

    @staticmethod
    def _create_block(
        in_dim: int, out_dim: int, kernel_size: int, stride: int, dropout: float
    ):
        block = nn.Sequential(
            nn.Conv2d(
                in_dim, out_dim, kernel_size=kernel_size, stride=stride,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        return block

    @staticmethod
    def split_output(out):
        is_object = out[:, 0, ...]
        coords = out[:, [1, 2], ...]
        dims = out[:, [3, 4], ...]
        class_logits = out[:, 5:, ...]
        return is_object, coords, dims, class_logits


def loss_func(
    y_preds,
    y_true,
    pos_weight: float = 1.0,
    is_object_weight: float = 1.0,
    center_coord_weight: float = 1.0,
    frame_dims_weight: float = 1.0,
    class_prob_weight: float = 1.0,
) -> torch.Tensor:
    true_is_object, true_coords, true_dims, true_class_logits = Model.split_output(
        y_true
    )
    preds_is_object, preds_coords, preds_dims, preds_class_logits = Model.split_output(
        y_preds
    )

    # Note: weight for detecting objects
    pos_weight = torch.tensor([pos_weight], device=y_preds.device)

    # Note: no need to shrink 'preds_is_object' because loss function
    # (bce loss w/ logits) already incorporate the sigmoid function.
    total_loss = is_object_weight * nn.functional.binary_cross_entropy_with_logits(
        preds_is_object, true_is_object, pos_weight=pos_weight,
    )

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

    total_loss += center_coord_weight * nn.functional.binary_cross_entropy_with_logits(
        preds_coords, true_coords
    )

    preds_dims = torch.exp(preds_dims)  # Note: ensure > 0
    total_loss += frame_dims_weight * nn.functional.mse_loss(preds_dims, true_dims)

    preds_class_logits = preds_class_logits.view(-1, config.NUM_CLASSES)

    # Note: no need to softmax class logits since the loss functiona
    # (cross entropy) already incorporate the (log-)softmax function.
    total_loss += class_prob_weight * nn.functional.cross_entropy(
        preds_class_logits, true_class_logits
    )

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
    train_epochs = 15
    epochs_per_checkpoint = 5
    device = "cuda"
    train_batch_size = 48
    eval_batch_size = 16
    checkpoint_path = "dl_checkpoint.tar"
    num_eval_inst = 5
    dropout = 0.4

    train_dataset, eval_dataset = utils.get_data(train_frac=0.95)

    model = Model(dropout=dropout)
    optim = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, factor=0.9, patience=5, verbose=True
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

    model = model.to(device)

    if train_epochs > 0:
        criterion = functools.partial(
            loss_func,
            pos_weight=5.0,
            is_object_weight=2.5,
            center_coord_weight=1.25,
            frame_dims_weight=1.0,
            class_prob_weight=1.0,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, batch_size=train_batch_size
        )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=eval_batch_size
        )

        for epoch in range(1, 1 + train_epochs):
            print(f"Epoch: {epoch:4d} / {train_epochs}...")
            model.train()
            train_total_batches = 0
            train_loss = train_detection_acc = 0.0

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
                    train_detection_acc += (
                        (
                            (torch.sigmoid(y_preds[:, 0]) >= 0.6)
                            == (y_batch[:, 0] >= 0.999)
                        )
                        .float()
                        .mean()
                        .item()
                    )

                del X_batch, y_batch

            model.eval()
            eval_total_batches = 0
            eval_loss = eval_detection_acc = 0.0

            for X_batch, y_batch in tqdm.auto.tqdm(eval_dataloader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_preds = model(X_batch)
                loss = criterion(y_preds, y_batch)

                eval_total_batches += 1
                eval_loss += loss.item()
                eval_detection_acc += (
                    ((torch.sigmoid(y_preds[:, 0]) >= 0.6) == (y_batch[:, 0] >= 0.999))
                    .float()
                    .mean()
                    .item()
                )

                del X_batch, y_batch

            train_loss /= train_total_batches
            train_detection_acc /= train_total_batches

            eval_loss /= eval_total_batches
            eval_detection_acc /= eval_total_batches

            scheduler.step(eval_loss)

            print(
                f"train loss: {train_loss:.4f} - train detection acc: {train_detection_acc:.4f}"
            )
            print(
                f"eval  loss: {eval_loss:.4f} - eval  detection acc: {eval_detection_acc:.4f}"
            )

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

    X_batch = eval_dataset.tensors[0]
    insts_eval = X_batch[:num_eval_inst]
    y_preds = predict(model, insts_eval.to(device))

    for inst, pred in zip(insts_eval, y_preds):
        print(pred[0, ...].max().item())
        utils.plot_instance(inst.detach().cpu().squeeze(), pred.detach().cpu())

    print("Done.")


if __name__ == "__main__":
    _test()
