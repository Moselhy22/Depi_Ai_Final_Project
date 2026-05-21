import os
from icrawler.builtin import BingImageCrawler

BASE_DIR = r"D:\DEPI GP\data\destraction detection data"

classes = {
    "phone": [
        "driver using mobile phone while driving",
        "texting while driving car",
        "talking on phone driving",
        "hand holding phone in car"
    ],
    "cigarette": [
        "driver smoking cigarette in car",
        "person smoking while driving",
        "cigarette in hand driving car"
    ],
    "food": [
        "eating while driving car",
        "driver eating burger in car",
        "driver eating snacks while driving"
    ],
    "drink": [
        "drinking water while driving",
        "driver drinking coffee in car",
        "holding cup while driving"
    ],
    "headphones": [
        "driver wearing headphones in car",
        "person wearing headset driving",
        "headphones on driver in vehicle"
    ],
    "earphones": [
        "driver wearing earphones in car",
        "person using earbuds while driving",
        "earphones in ear driving car"
    ]
}

NUM_IMAGES = 250

for cls, keywords in classes.items():
    path = os.path.join(BASE_DIR, cls)
    os.makedirs(path, exist_ok=True)

    for kw in keywords:
        crawler = BingImageCrawler(storage={"root_dir": path})
        crawler.crawl(keyword=kw, max_num=NUM_IMAGES)

print("DONE")