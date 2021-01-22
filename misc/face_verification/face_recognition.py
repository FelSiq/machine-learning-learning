import typing as t
import glob
import pickle

import PIL
import torch.nn as nn
import facenet_pytorch
import re
import numpy as np

RECOVER_NAME_REG = re.compile(r"(?<=face_database/)(.+)(?=\.jpg)")


def prepare_img(filepath: str, return_name: bool = False, save_cropped: bool = False):
    who = RECOVER_NAME_REG.search(filepath)

    if who:
        who = who.group(1)

    img = PIL.Image.open(filepath)
    cropper = facenet_pytorch.MTCNN(image_size=96, margin=0)
    save_path = f"cropped/{who}_cropped.jpg" if save_cropped and who else None
    cropped_img = cropper(img, save_path=save_path)
    cropped_img = cropped_img.unsqueeze(0)

    if return_name:
        return cropped_img, who

    return cropped_img


def build_database(model: nn.Module):

    model = model.eval()

    names = []
    embeddings = []

    for filepath in glob.iglob("face_database/*.jpg"):
        img, who = prepare_img(filepath, return_name=True, save_cropped=True)
        embed = model(img)

        names.append(who)
        embeddings.append(embed.detach().numpy())

    embeddings = np.vstack(embeddings)

    precomp_emb = {
        "names": names,
        "embeddings": embeddings,
    }

    with open("precomp_emb.pickle", "wb") as f_out:
        pickle.dump(precomp_emb, f_out, protocol=pickle.HIGHEST_PROTOCOL)

    return precomp_emb


def face_recognition(
    model: nn.Module,
    precomp_emb: np.ndarray,
    filepath: str,
    threshold: float = 0.7,
    verbose: bool = True,
):
    model.eval()

    img = prepare_img(filepath)
    embed = model(img).squeeze().detach().numpy()
    distances = np.linalg.norm(precomp_emb["embeddings"] - embed, axis=1)

    ind = distances.argmin()

    if verbose:
        print("Minimal distance:", distances[ind], f"(threshold is {threshold:.02f})")
        print("Who?             ", precomp_emb["names"][ind])
        print("Verdict:", end=" ")

    if distances[ind] > threshold:
        print("Face not recognized. Pleasy try again.")
        return False

    who = precomp_emb["names"][ind]
    print(f"Welcome, {who}!")

    return True


def _test():
    threshold = 0.8
    model = facenet_pytorch.InceptionResnetV1(pretrained="vggface2").eval()

    try:
        with open("precomp_emb.pickle", "rb") as f_in:
            precomp_emb = pickle.load(f_in)

        print("Loaded face database precomputed embeddings.")

    except FileNotFoundError:
        precomp_emb = build_database(model)

    for filepath in glob.iglob("face_tests/*.jpg"):
        print(40 * "=")
        print("Processing:", filepath)
        face_recognition(model, precomp_emb, filepath, threshold)
        print(40 * "=")
        print()


if __name__ == "__main__":
    _test()
