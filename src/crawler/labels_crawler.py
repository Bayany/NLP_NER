
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
from pandas import DataFrame as df

http = requests.Session()
http.mount("https://", HTTPAdapter(max_retries=5))
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}


COOKIES = {"pagGDPR": "true"}


def get_labels(type):
    url = f"https://purple-hyacinth.fandom.com/wiki/Category:{type}"
    resp = http.get(url, cookies=COOKIES,headers=headers)
    soup = BeautifulSoup(resp.content, features="lxml")
    members = soup.findAll("li", {"class": "category-page__member"})

    entities=[]
    for member in members:
        entities.append(member.a.get("title"))

    return entities


if __name__=="__main__":
    characters = get_labels("Characters")
    person_names =[]
    for character in characters:
        for c in character.split():
            if(len(c)>3):person_names.append(c)
    characters_df = df({ "Label": ["PERSON"] * len( person_names ),"Entity": person_names })
    
    locations = get_labels("Locations")

    locations_df = df({ "Label": ["LOC"] * len( locations ),"Entity": locations })
    labels_df= pd.concat([characters_df,locations_df])
    labels_df.to_csv("data/clean/labels.csv", index=False)
    print("DONE")