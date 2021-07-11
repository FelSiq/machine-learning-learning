import numpy as np
import bs4
import requests
import regex as re
import shutil
import tqdm.auto


re_spaces = re.compile(r"(?<=\>)\s+|\s+(?=\<)")


def _test():
    with open("captions.html", "r") as f:
        html = f.read()

    soup = bs4.BeautifulSoup(html, "html.parser")
    all_blocks = soup.find_all("tr")

    n = len(all_blocks)

    assert n % 2 == 0

    all_urls = all_blocks[0::2]
    all_desc = all_blocks[1::2]

    i = 0

    pbar = tqdm.auto.tqdm(zip(all_urls, all_desc), total=n // 2)

    for j, (url_img, descs) in enumerate(pbar, 1):
        url_img = url_img.find("a")
        descs = descs.find("ul")

        if not url_img:
            continue

        url_download = url_img.get("href") + "/sizes/s/"

        new_html = requests.get(url_download).text
        new_html = re_spaces.sub("", new_html)

        new_parser = bs4.BeautifulSoup(new_html, "html.parser")
        img_small_url = new_parser.find(
            "a", text="Download the Small 240 size of this photo"
        )

        if not img_small_url:
            continue

        img_small_url = img_small_url.get("href")
        img = requests.get(img_small_url, stream=True)

        if img.status_code != 200:
            continue

        with open(f"./images/{i}.jpg", "wb") as f_out:
            img.raw.decode_content = True
            shutil.copyfileobj(img.raw, f_out)

        with open(f"./descriptions/{i}.txt", "w") as f_out:
            f_out.write(descs.text.lstrip())

        i += 1
        pbar.set_description(f"Success rate: {100. * i / j:.2f}% ({i} of {j} so far)")


if __name__ == "__main__":
    _test()
