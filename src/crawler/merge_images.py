from glob import glob
import os
import cv2
import shutil


def has_blackpixels(img):
    bp_n = 0
    has_255_in_left = False
    for i in range(0, len(img)):

        if(has_255_in_left and img[i] < 10):
            bp_n += 1
        if((has_255_in_left and img[i] == 255) and bp_n > 0):
            has_255_in_left = False
        elif(img[i] == 255):
            has_255_in_left = True

    return bp_n > 0


def merge_chapter(chapter_no, base_folder):
    chapter_folder = f"{base_folder}/ch{chapter_no}"
    output_folder = f"{base_folder}/merged/ch{chapter_no}"
    if not os.path.isdir(output_folder ):
        os.makedirs(output_folder )

    pages_need_merge = {}
    l = len(glob(f"{chapter_folder}/*.jpg"))
    print(chapter_folder)
    for j in range(1, l):
        im = cv2.imread(f"{chapter_folder}/p{j}.jpg", cv2.IMREAD_GRAYSCALE)
        has_top = has_blackpixels(im[0])
        has_bottom = has_blackpixels(im[-1])
        if(has_top or has_bottom):
            pages_need_merge[j] = (has_top, has_bottom)

    page_count = 1
    curr_page = 1
    while curr_page < l:
        img_paths = ""
        while((curr_page in pages_need_merge and pages_need_merge[curr_page][1])
              and (curr_page+1 in pages_need_merge and pages_need_merge[curr_page+1][0])
              ):
            img_paths += f" {chapter_folder}/p{curr_page}.jpg"
            curr_page += 1
        if(img_paths):
            img_paths += f" {chapter_folder}/p{curr_page}.jpg"
            os.system(
                f"convert -append {img_paths} {output_folder}/p{page_count}.jpg")
        else:
            shutil.copyfile(
                f"{chapter_folder}/p{curr_page}.jpg", f"{output_folder}/p{page_count}.jpg")
        curr_page += 1
        page_count += 1
