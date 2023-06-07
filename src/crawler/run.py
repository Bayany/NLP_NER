#@title Load EasyOCR
import easyocr
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory


from glob import glob
import time
import os

from webtoondl import download_chapter
from merge_images import merge_chapter
import PIL.Image

if not hasattr(PIL.Image, 'Resampling'):  # Pillow<9.0
  PIL.Image.Resampling = PIL.Image


base_path="data/raw/temp/PurpleHyacinth"


if __name__ == "__main__":
    url = "https://www.webtoons.com/en/mystery/purple-hyacinth/list?title_no=1621"
    for i in range(1,100):

        if(not os.path.exists(f"{base_path}/ch{i}/")):
          download_chapter(1621, i, base_path)
        if(not os.path.exists(f"{base_path}/merged/ch{i}/")):
          merge_chapter(i, base_path)

        chapter_folder=f"{base_path}/merged/ch{i}/"
        begin= time.time()
        os.mkdir("data/raw/PurpleHyacinth/")
        with open(f"data/raw/PurpleHyacinth/ch{i}.txt","w") as f:
            l=len(glob(f"{chapter_folder}/*.jpg"))
            for j in range(1,l):
                result = reader.readtext(f"{chapter_folder}/p{j}.jpg")
                text=" ".join(r[1] for r in result).capitalize()
                if(text):
                  f.write(text+"\n")
        end = time.time()
        print(f"{int((end-begin)//60)} min : {int((end-begin)%60)} sec")