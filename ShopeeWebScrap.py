import httpx
from rich import print
import pandas as pd
import json


def download_json(url):
    resp = httpx.get(url)
    try:
        for node in resp.json()['data']['ratings']:
            yield node
    except (json.JSONDecodeError, KeyError):
        pass  # skip this iteration if the response is not valid JSON or doesn't contain the expected data

def save_to_csv(data):
    df = pd.DataFrame(data)
    df.to_csv('results.csv', index=False)

def save_to_json(data):
    with open('results.json','w') as f:
        json.dump(data,f)


def main():

    results = []
    x = 0
    for i in range(0,50):
        url = f"https://shopee.ph/api/v2/item/get_ratings?exclude_filter=0&filter=0&filter_size=0&flag=1&fold_filter=0&itemid=6345876261&limit=6&offset={x}&relevant_reviews=false&request_source=1&shopid=171457021&tag_filter=&type=0&variation_filters=" 
        for item in download_json(url):
            results.append(item['comment'])
            x+=6

    save_to_json(results)
    save_to_csv(results)

if __name__ == "__main__":
    main()