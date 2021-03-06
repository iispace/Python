## The source of this code comes from 'https://dev.to/dmitryzub/scrape-google-scholar-with-python-32oh'
## A slight modification has been done to separately get the results provided in a form of a PDF.

import requests, lxml, os, json, numpy as np
from bs4 import BeautifulSoup

proxies = {
    'http': os.getenv('HTTP_PROXY') 
}

headers = {
    'User-agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582"
}

year_to_start = 2021   # year to start search
search_keyword = "검색 주제어"
search_language = "ko"  # or "en"
max_start = 10

htmls = []
soups = []

for i in range(0, max_start):
    params = {
        "as_ylo": year_to_start,
        "q": search_keyword,
        "hl": search_language,
        "start": i*10
    }
    print(f'scrapping from the start of : {params["start"]}')
    html = requests.get('https://scholar.google.com/scholar', headers=headers, params=params, proxies=proxies).text
    soup = BeautifulSoup(html, 'lxml')
    htmls.append(html)
    soups.append(soup)
    
data = []
pdfs = []

for soup in soups:
    for result in soup.select('.gs_ri'):
        title = result.select_one('.gs_rt').text
        title_link = result.select_one('.gs_rt a')['href']
        publication_info = result.select_one('.gs_a').text

        data.append({
        'title': title,
        'title_link': title_link,
        'publication_info': publication_info,
        })

        # pdf로 직접 제공되는 논문들 추출
        suffixes = ("pdf","page")

        if title_link.endswith(suffixes):
            suffix = title_link.split('.')[len(title_link.split('.'))-1]
            download_link = title_link.replace(suffix, "pdf")
            pdfs.append({
                'title': title,
                'title_link': title_link,
                'download_link': download_link,
                'publication_info': publication_info,
            })

# 검색된 모든 결과 출력
for i, datum in zip(np.arange(0, len(data)), data):
    print(f'[{i}]: {datum["title"]}, {datum["title_link"]}, ({datum["publication_info"]})')

# 검색 결과 중에서 PDF로 제공되는 논문(제목, 링크, 저자 정보)만 출력
for i, pdf in zip(np.arange(0, len(pdfs)) , pdfs):
    print(f'[{i}]: {pdf["title"]}, {pdf["download_link"]}, ({pdf["publication_info"]})')
