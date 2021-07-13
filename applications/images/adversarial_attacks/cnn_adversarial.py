import PIL
import tqdm.auto
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import itertools


def white_box_adversarial(
    desired_class: int = 255, train_epochs: int = 2000, device: str = "cuda"
):
    """White box: access to the model gradients."""
    assert 0 <= desired_class < 1000

    lr = 2e-3
    filename = "bird.jpg"

    model = torchvision.models.googlenet(pretrained=True)
    model = model.eval().to(device)

    input_image = PIL.Image.open(filename)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    X = transform(input_image).unsqueeze(0)
    X_orig = torch.clone(X).to(device)

    X = torch.autograd.Variable(X.to(device), requires_grad=True)
    y_adv = torch.zeros((1, 1000), dtype=torch.float)
    y_adv[0, desired_class] = 1.0
    y_adv = y_adv.to(device)

    orig_y_pred = model(X).squeeze().detach().cpu().numpy()
    orig_cls_id = orig_y_pred.argmax()
    orig_pred_conf = orig_y_pred[orig_cls_id]

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()
    optim = torch.optim.Adam([X], lr=lr)

    pbar = tqdm.auto.tqdm(np.arange(1, 1 + train_epochs))

    for epoch in pbar:
        optim.zero_grad()
        y_pred = model(X)
        loss_a = criterion_bce(y_pred, y_adv)
        loss_b = criterion_mse(X, X_orig)
        loss = loss_a + loss_b
        loss.backward()
        optim.step()
        pbar.set_description(f"epoch: {epoch:<8}, loss: {loss.item():.4f}")

    adv_y_pred = model(X).squeeze().cpu().detach().numpy()
    adv_cls_id = adv_y_pred.argmax()
    adv_pred_conf = adv_y_pred[adv_cls_id]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10))
    fig.suptitle(f"Desired class id: {desired_class}")
    ax1.imshow(X_orig.squeeze().permute(1, 2, 0).cpu().numpy())
    ax1.set_title(f"Original class id: {orig_cls_id} (conf: {orig_pred_conf:.3f})")
    ax2.imshow(X.squeeze().permute(1, 2, 0).cpu().detach().numpy())
    ax2.set_title(f"Adversarial class id: {adv_cls_id} (conf: {adv_pred_conf:.3f})")
    plt.show()


def _test():
    # white_box_adversarial()
    black_box_adversarial()


if __name__ == "__main__":
    _test()
