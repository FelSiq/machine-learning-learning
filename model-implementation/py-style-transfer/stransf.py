import typing as t
import os

import torch
import torch.nn as nn
import torchvision
import numpy as np
import imageio


_LAYER_ACT_GENERATED = []  # type: t.List[torch.Tensor]
_LAYER_ACT_STYLE = []  # type: t.List[torch.Tensor]
_LAYER_ACT_CONTENT = []  # type: t.List[torch.Tensor]


def content_cost_fn(a_content, a_generated):
    scale = 0.25 / np.prod(a_content.shape.tolist())
    content_cost = scale * torch.sum(torch.square(a_content - a_generated))
    return content_cost


def gram_matrix(mat):
    return torch.matmul(mat, mat.transpose(-1, -2))


def layer_style_cost_fn(layer_gram_mat_style, a_generated):
    a_generated_shape = gram_mat_style.shape.tolist()

    scale = 0.25 / np.square(np.prod(a_generated_shape))

    _, dim_h, dim_w, dim_c = a_generated_shape

    a_generated = a_generated.permute(0, 3, 1, 2).view((dim_c, dim_h * dim_w))

    gram_mat_generated = gram_matrix(a_generated)

    style_cost = scale * torch.sum(
        torch.square(layer_gram_mat_style - gram_mat_generated)
    )

    return style_cost


def full_style_cost_fn(gram_mats_style, a_generated, style_weights: t.Sequence[float]):
    total_style_loss = 0.0

    for i, style_weight in enumerate(style_weights):
        total_style_loss = style_weight * layer_style_cost_fn(
            gram_mats_style[i], a_generated
        )

    return total_style_loss


def full_cost_fn(
    a_content,
    a_generated,
    gram_mats_style,
    style_weights: t.Sequence[float],
    style_weight: float = 4,
):
    total_cost = content_cost_fn(
        a_content, a_generated
    ) + style_weight * full_style_cost_fn(gram_mat_style, a_generated, style_weights)
    return total_cost


def hook_extract_activation_fn(act_container: t.List[torch.Tensor], l_ind: int):
    def hook(model, input, output):
        if len(act_container) <= l_ind:
            act_container.extend((l_ind - len(act_container) + 1) * [None])

        act_container[l_ind] = output.detach()

    return hook


def add_hooks(model, act_container, conv_layer_inds):
    hooks = []

    for i, l_ind in enumerate(sorted(conv_layer_inds)):
        hook_fn = hook_extract_activation_fn(act_container, i)
        hook = model[l_ind].register_forward_hook(hook_fn)
        hooks.append(hook)

    return hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def get_model(img_content, img_style, user_defined_style_layers, device: str):
    model = torchvision.models.vgg16(pretrained=True).features

    model.eval()

    model = model.to(device)

    conv_layer_inds = [i for i, l in enumerate(model) if isinstance(l, nn.Conv2d)]

    conv_layer_inds = [conv_layer_inds[k] for k in user_defined_style_layers]

    hooks = add_hooks(model, _LAYER_ACT_STYLE, conv_layer_inds)
    model(img_style)
    remove_hooks(hooks)

    add_hooks(model, _LAYER_ACT_CONTENT, conv_layer_inds)
    model(img_content)
    remove_hooks(hooks)

    add_hooks(model, _LAYER_ACT_CONTENT, conv_layer_inds)

    del conv_layer_inds

    assert len(_LAYER_ACT_STYLE) == len(user_defined_style_layers)
    assert len(_LAYER_ACT_CONTENT) == len(user_defined_style_layers)

    return model


def read_image(path: str, device: str):
    img = torch.from_numpy(imageio.imread(path))
    img = img.float()
    img = img.permute(2, 0, 1)  # Note: set channels first

    # Note: preprocessing required by pretrained torch VGG16
    normalizer = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    img = normalizer(img)
    img = img.unsqueeze(0)  # Note: add batch dimension
    img = img.to(device)

    assert img.ndim == 4

    return img


def generate_output(generated_img):
    generated_img = generated_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    generated_img = 255 * (generated_img - generated_img.max()) / np.ptp(generated_img)
    return generated_img.astype(np.uint8)


def train(model, img_content, img_style, train_it_num: int):
    generated_img = torch.randn_like(img_content, requires_grad=True)

    for i in np.arange(train_it_num):
        pass

    return generate_output(generated_img)


def _test():
    import argparse

    parser = argparse.ArgumentParser(description="Neural style transfer")
    parser.add_argument(
        "content_image_path", type=str, help="Path to the content image."
    )
    parser.add_argument("style_image_path", type=str, help="Path to the style image.")
    parser.add_argument(
        "--num-train-it",
        type=int,
        nargs=1,
        default=10,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--style-layer-indices",
        nargs="+",
        type=int,
        default=(6, 7, 8),
        help="Indices of VGG16 Conv2d indices to use as style layers. Must be in {0, ..., 13}.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Ig given, run in CPU rather than GPU",
    )
    args = parser.parse_args()

    assert all(
        map(lambda l: 0 <= l <= 13, args.style_layer_indices)
    ), "Style layer indices must be in {0, ..., 13}"

    device = "cpu" if args.cpu else "cuda"

    img_content = read_image(args.content_image_path, device)
    img_style = read_image(args.style_image_path, device)

    model = get_model(
        img_content, img_style, frozenset(args.style_layer_indices), device
    )

    generated_img = train(model, img_content, img_style, args.num_train_it)

    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir_path = os.path.join(cur_dir_path, "output")

    try:
        os.mkdir(output_dir_path)
        print(f"Created ouput directory: {output_dit_path}")

    except FileExistsError:
        pass

    imageio.imsave(os.path.join(output_dir_path, "test.jpg"), generated_img)


if __name__ == "__main__":
    _test()
