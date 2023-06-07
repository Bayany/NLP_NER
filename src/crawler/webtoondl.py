
import os
import tqdm
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter, Retry

http = requests.Session()
http.mount("https://", HTTPAdapter(max_retries=5))


COOKIES = {"pagGDPR": "true"}


def get_filetype(url):
    if url.endswith("/"):
        return url[:-1].split("/")[-1].split("?")[0].split(".")[-1]
    else:
        return url.split("/")[-1].split("?")[0].split(".")[-1]


def download_chapter(webtoon_id, chapter_no, output_folder):
    path = f"{output_folder}/ch{chapter_no}/"
    if not os.path.isdir(path):
        os.makedirs(path)

    url = f"https://webtoons.com/en/mystery/purple-hyacinth/a/viewer?title_no={webtoon_id}&episode_no={chapter_no}"
    resp = http.get(url, cookies=COOKIES)
    url = resp.url
    path += "p"
    soup = BeautifulSoup(resp.content, features="lxml")
    content = soup.find("div", {"id": "content"})
    imagelist = content.find("div", {"id": "_imageList"})
    images = []
    if imagelist is not None:
        for img in imagelist.findAll("img", {"class": "_images"}):
            images.append(img["data-url"])

    page_no = 1
    for image_url in tqdm.tqdm(images):
        ext = get_filetype(image_url)
        resp = http.get(image_url, headers={"Referer": url})
        assert resp.status_code == 200
        pic = f"{path}{page_no}.{ext}"
        with open(pic, "wb") as file:
            file.write(resp.content)
        if(ext == "gif"):
            os.system(f"convert -coalesce {pic} {pic[:-3]}jpg")
            os.remove(pic)
        page_no += 1
