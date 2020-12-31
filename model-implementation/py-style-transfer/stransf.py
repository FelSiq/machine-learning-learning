import typing as t
import os

import torch
import torch.nn as nn
import torchvision
import numpy as np
import imageio


_MAX_LAYER_INDEX = 12
_LAYER_ACT_GENERATED_CONTENT = []  # type: t.List[torch.Tensor]
_LAYER_ACT_GENERATED_STYLES = []  # type: t.List[torch.Tensor]
_LAYER_GRAM_MAT_STYLE = []  # type: t.List[torch.Tensor]
_LAYER_ACT_CONTENT = []  # type: t.List[torch.Tensor]


def content_cost_fn(a_content: torch.Tensor, a_generated: torch.Tensor):
    scale = 0.25 / a_content.numel()
    content_cost = scale * torch.sum(torch.square(a_content - a_generated))
    return content_cost


def gram_matrix(mat: torch.Tensor) -> torch.Tensor:
    _, dim_c, dim_h, dim_w = mat.shape
    mat = mat.view((dim_c, dim_h * dim_w))
    gram_mat = torch.matmul(mat, mat.T)
    return gram_mat


def layer_style_cost_fn(layer_gram_mat_style: torch.Tensor, a_generated: torch.Tensor):
    scale = 0.25 / (layer_gram_mat_style.numel() ** 2)

    gram_mat_generated = gram_matrix(a_generated)

    style_cost = scale * torch.sum(
        torch.square(layer_gram_mat_style - gram_mat_generated)
    )

    return style_cost


def full_style_cost_fn(
    gram_mats_style: t.List[torch.Tensor],
    a_generated: t.List[torch.Tensor],
    style_weights: t.List[float],
):
    total_style_loss = 0.0

    for i, style_rel_weight in enumerate(style_weights):
        total_style_loss = style_rel_weight * layer_style_cost_fn(
            gram_mats_style[i], a_generated[i]
        )

    return total_style_loss


def full_cost_fn(
    a_content: torch.Tensor,
    a_generated_content: torch.Tensor,
    a_generated_styles: t.List[torch.Tensor],
    gram_mats_style: t.List[torch.Tensor],
    style_weights: t.List[float],
    style_rel_weight: float,
):
    total_cost = content_cost_fn(
        a_content, a_generated_content
    ) + style_rel_weight * full_style_cost_fn(
        gram_mats_style, a_generated_styles, style_weights
    )
    return total_cost


def hook_extract_activation_fn(act_container: t.List[torch.Tensor], l_ind: int):
    def hook(model, input, output):
        if len(act_container) <= l_ind:
            act_container.extend((l_ind - len(act_container) + 1) * [None])

        act_container[l_ind] = output

    return hook


def add_hooks(
    model: nn.ParameterList,
    act_container: t.List[torch.Tensor],
    conv_layer_inds: t.Union[int, t.List[int]],
) -> t.List[t.Any]:
    if isinstance(conv_layer_inds, int):
        conv_layer_inds = [conv_layer_inds]

    hooks = []

    for i, l_ind in enumerate(sorted(conv_layer_inds)):
        hook_fn = hook_extract_activation_fn(act_container, i)
        hook = model[l_ind].register_forward_hook(hook_fn)
        hooks.append(hook)

    return hooks


def fill_act_global_lists(
    model: nn.ParameterList,
    input: torch.Tensor,
    container: t.List[torch.Tensor],
    inds: t.Union[int, t.List[int]],
) -> None:
    hooks = add_hooks(model, container, inds)
    model(input)

    for hook in hooks:
        hook.remove()

    del hooks


def get_model(
    img_content: torch.Tensor,
    img_style: torch.Tensor,
    user_defined_content_layer: int,
    user_defined_style_layers: t.Iterable[int],
    device: str,
) -> nn.ParameterList:
    # Note: load VGG16 pretrained model
    model = torchvision.models.vgg16(pretrained=True)
    model.eval()
    model = model.features
    model = model.to(device)

    # Note: cast user-defined conv layer indices to real conv layer indices
    all_act_layer_inds = [i for i, l in enumerate(model) if isinstance(l, nn.ReLU)]
    user_defined_style_layers = sorted(set(user_defined_style_layers))
    conv_layer_inds = [all_act_layer_inds[k] for k in user_defined_style_layers]

    # Note: get style image activations
    fill_act_global_lists(
        model, img_style, _LAYER_GRAM_MAT_STYLE, user_defined_style_layers
    )

    # Note: cast style image activations to its Gram Matrix
    for i, act in enumerate(_LAYER_GRAM_MAT_STYLE):
        _LAYER_GRAM_MAT_STYLE[i] = gram_matrix(act)

    # Note: get content image activations
    fill_act_global_lists(
        model, img_content, _LAYER_ACT_CONTENT, user_defined_content_layer
    )

    # Note: add hooks to recover generated image's activations
    add_hooks(model, _LAYER_ACT_GENERATED_STYLES, user_defined_style_layers)
    add_hooks(model, _LAYER_ACT_GENERATED_CONTENT, user_defined_content_layer)

    # Note: clean-up and sanity checks
    del conv_layer_inds
    assert len(_LAYER_GRAM_MAT_STYLE) == len(user_defined_style_layers)
    assert len(_LAYER_ACT_CONTENT) == 1
    assert not _LAYER_ACT_GENERATED_CONTENT
    assert not _LAYER_ACT_GENERATED_STYLES

    return model


def read_image(
    path: str, device: str, shape: t.Optional[t.Tuple[int, int]] = None
) -> torch.Tensor:
    img = imageio.imread(path)
    img = torch.from_numpy(img)
    img = img.float()
    img = img.permute(2, 1, 0)  # Note: set channels first
    img = img.unsqueeze(0)  # Note: add batch dimension

    if shape is not None:
        img = nn.functional.interpolate(img, size=tuple(shape))

    img = img.to(device)

    assert img.dim() == 4

    return img


def generate_output_img(generated_img: torch.Tensor) -> np.ndarray:
    # Note: (channels first -> channels last) + (tensor -> numpy array)
    gen_img_np = generated_img.squeeze().permute(2, 1, 0).detach().cpu().numpy()

    # Note: clip in-place to range [0, 255]
    gen_img_np.clip(0.0, 255.0, out=gen_img_np)

    return gen_img_np.astype(np.uint8)


def init_generated_img(
    base_image: torch.Tensor, init_noise_ratio: float
) -> torch.Tensor:
    assert 0.0 <= init_noise_ratio <= 1.0

    base_init_img = base_image.contiguous()

    generated_img = (
        1.0 - init_noise_ratio
    ) * base_init_img + 255.0 * init_noise_ratio * torch.rand_like(base_image)

    generated_img.requires_grad = True

    del base_init_img

    return generated_img


def train(
    model: nn.ParameterList,
    img_content: torch.Tensor,
    img_style: torch.Tensor,
    train_it_num: int,
    style_weights: t.List[float],
    style_rel_weight: float,
    init_noise_ratio: float,
    init_from_style: bool,
    it_to_print: int = 100,
) -> torch.Tensor:
    assert np.isclose(1.0, sum(style_weights)), "Style weights do not sum to 1.0"
    assert style_rel_weight >= 0.0, "Style relative weight must be non-negative"
    assert len(style_weights) == len(_LAYER_GRAM_MAT_STYLE)

    generated_img = init_generated_img(
        img_style if init_from_style else img_content, init_noise_ratio
    )

    optim = torch.optim.Adam([generated_img], lr=0.2)
    print_fill = len(str(train_it_num))

    for i in np.arange(1, 1 + train_it_num):
        optim.zero_grad()

        # Note: the actual output does not matter since the activations
        # will be stored in global lists with previously registered hooks.
        model(generated_img)

        loss = full_cost_fn(
            a_content=_LAYER_ACT_CONTENT[0],
            a_generated_content=_LAYER_ACT_GENERATED_CONTENT[0],
            a_generated_styles=_LAYER_ACT_GENERATED_STYLES,
            gram_mats_style=_LAYER_GRAM_MAT_STYLE,
            style_weights=style_weights,
            style_rel_weight=style_rel_weight,
        )

        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(generated_img, max_norm=5.0)
        optim.step()

        if it_to_print > 0 and i % it_to_print == 0:
            print(
                f"Iteration: {i:<{print_fill}} / {train_it_num} - loss: {loss.item():.4f}"
            )

    return generate_output_img(generated_img)


def save_generated_img(
    generated_img: np.ndarray,
    output_name: str,
    output_format: str,
    output_dir: str = "output",
):
    assert generated_img.ndim == 3

    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    output_dir_path = os.path.join(cur_dir_path, output_dir)

    try:
        os.mkdir(output_dir_path)
        print(f"Created ouput directory: {output_dit_path}")

    except FileExistsError:
        pass

    if not output_format.startswith("."):
        output_format = "." + output_format

    if not output_name.endswith(output_format):
        output_name += output_format

    imageio.imwrite(
        os.path.join(output_dir_path, output_name),
        generated_img,
        format=output_format,
    )


def _test():
    import argparse

    parser = argparse.ArgumentParser(description="Neural style transfer")
    parser.add_argument(
        "content_image_path", type=str, help="Path to the content image."
    )
    parser.add_argument("style_image_path", type=str, help="Path to the style image.")
    parser.add_argument(
        "--train-it-num",
        type=int,
        default=512,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="generated",
        help="Name of the generated output image.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="jpg",
        help="File type of the generated output image.",
    )
    parser.add_argument(
        "--content-layer-index",
        type=int,
        default=9,
        help=f"Index of VGG16 Conv2d layers to use as content layer. Must be in {{0, ..., {_MAX_LAYER_INDEX}}}.",
    )
    parser.add_argument(
        "--style-rel-weight",
        type=float,
        default=32.0,
        help="Weight of style loss relative to the content loss. (a.k.a. 'beta'/'alpha' from the original paper)",
    )
    parser.add_argument(
        "--init-noise-ratio",
        type=float,
        default=0.3,
        help="Ratio of noise and content image interpolation while initializing generated image.",
    )
    parser.add_argument(
        "--style-layer-inds",
        nargs="+",
        type=int,
        default=(6, 8, 9, 10, 12),
        help=f"Indices of VGG16 Conv2d layers to use as style layers. Must be in {{0, ..., {_MAX_LAYER_INDEX}}}.",
    )
    parser.add_argument(
        "--style-layer-weights",
        nargs="+",
        type=float,
        default=(0.05, 0.10, 0.25, 0.35, 0.25),
        help="Weights for each Conv2d layer. Number of args must match '--style-layer-inds'.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="If given run in CPU rather than GPU",
    )
    parser.add_argument(
        "--init-from-style",
        action="store_true",
        help="If given init the generated image from the style image rather than the content image.",
    )
    args = parser.parse_args()

    assert all(
        map(lambda l: 0 <= l <= _MAX_LAYER_INDEX, args.style_layer_inds)
    ), f"Style layer inds must be in {{0, ..., {_MAX_LAYER_INDEX}}}"

    assert (
        0 <= args.content_layer_index <= _MAX_LAYER_INDEX
    ), f"Content layer index must be in {{0, ..., {_MAX_LAYER_INDEX}}}"
    assert args.train_it_num > 0

    device = "cpu" if args.cpu else "cuda"

    img_content = read_image(args.content_image_path, device, shape=(900, 472))
    img_style = read_image(args.style_image_path, device, shape=img_content.shape[-2:])

    model = get_model(
        img_content,
        img_style,
        args.content_layer_index,
        args.style_layer_inds,
        device,
    )

    style_weights = args.style_layer_weights

    if style_weights is None:
        num_style_layers = len(args.style_layer_inds)
        style_weights = num_style_layers * [1.0 / num_style_layers]

    print(model, end="\n\n")
    print("Configuration:")
    print("Device chosen                    :", device)
    print("Style layer indices              :", args.style_layer_inds)
    print("Style layer weights              :", style_weights)
    print("Style relative weight to content :", args.style_rel_weight)
    print("Content layer index              :", args.content_layer_index)
    print("Init from style image            :", args.init_from_style)

    generated_img = train(
        model=model,
        img_content=img_content,
        img_style=img_style,
        train_it_num=args.train_it_num,
        style_weights=style_weights,
        style_rel_weight=args.style_rel_weight,
        init_noise_ratio=args.init_noise_ratio,
        init_from_style=args.init_from_style,
    )

    save_generated_img(generated_img, args.output_name, args.output_format)


if __name__ == "__main__":
    _test()
