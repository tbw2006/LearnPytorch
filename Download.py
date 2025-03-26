import os
import tarfile
import requests

# 下载数据集压缩包
url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
response = requests.get(url, stream=True)
with open("102flowers.tgz", "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

# 解压到指定目录
with tarfile.open("102flowers.tgz") as tar:
    tar.extractall(path="./data")
os.rename("./jpg", "./data/oxford_flowers102")  # 重命名文件夹