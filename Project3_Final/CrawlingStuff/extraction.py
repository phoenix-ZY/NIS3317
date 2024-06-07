from utils import *
from lxml import etree
from pprint import pprint
import os
import csv
from datetime import datetime
import pandas as pd
from time import sleep


def ajk_html_process(response, timestamp):
    if response.status_code==302 and response.headers.get('Location')=="https://www.anjuke.com/antispam-block/?from=antispam":
        return None
    rsp_rst = []

    html_content = response.content.decode('utf-8')
    tree = etree.HTML(html_content)
    if tree is None:
        print("[No Data in This Page]")
        return rsp_rst
    div_elements = tree.xpath('//*[@id="esfMain"]/section/section[3]/section[1]/section[2]/div')
    # //*[@id="esfMain"]/section/section[3]/section[1]/section[2]/div[1]/a

    for div in div_elements:
        tmp_link = div.xpath('./a/attribute::href')[0]
        id = re.compile(r'view/([A-Za-z0-9]+)\?').search(tmp_link).group(1)

        title = div.xpath('./a/div[2]/div[1]/div[1]/h3/text()')[0]

        room_element = div.xpath('./a/div[2]/div[1]/section/div[1]/p[1]/span')
        room_info = ''
        for room in room_element:
            room_info += room.xpath("text()")[0]

        area = div.xpath('./a/div[2]/div[1]/section/div[1]/p[2]/text()')[0].strip()

        try:
            direction = div.xpath('./a/div[2]/div[1]/section/div[1]/p[3]/text()')[0]
        except:
            direction=None

        try:
            floor = div.xpath('./a/div[2]/div[1]/section/div[1]/p[4]/text()')[0].strip()
        except:
            floor = None

        try:
            history = div.xpath('./a/div[2]/div[1]/section/div[1]/p[5]/text()')[0].strip()
        except:
            history=None

        name = div.xpath('./a/div[2]/div[1]/section/div[2]/p[1]/text()')[0]

        location_element = div.xpath('./a/div[2]/div[1]/section/div[2]/p[2]/span')
        location = ""
        for loc in location_element:
            location += loc.xpath('text()')[0] + ' '
        location = location[:-1]

        label_element = div.xpath('./a/div[2]/div[1]/section/div[3]/span')
        lables = []
        for lable in label_element:
            lables.append(lable.xpath("text()")[0])

        price_element = div.xpath('./a/div[2]/div[2]/p[1]/span')
        price = ''
        for i in price_element:
            price += i.xpath('text()')[0]

        avg_price = div.xpath('./a/div[2]/div[2]/p[2]/text()')[0]

        div_info = {'id': id, 'title': title, 'timestamp': timestamp, 'room_info': room_info, 'area': area, 'direction': direction, 'floor': floor, 'history': history, 'name': name, 'location': location, "labels": lables, 'price': price, "avg_price": avg_price}
        rsp_rst.append(div_info)
    print("[Collected",len(rsp_rst),"data]")
    if len(rsp_rst)==0:
        print(response.request.url)

    return rsp_rst


def save_formatted_data(file_path, data):
    if data is None:
        print("[Data is None, Saving Stopped]")
        return
    columns = ['id', 'name', 'location', 'room_info', 'floor', 'direction', 'area', 'price', 'avg_price', 'history', 'title', 'labels', 'timestamp']

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        if not file_exists:
            writer.writeheader()

        for row in data:
            writer.writerow(row)

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_filtered = df.loc[df.groupby('id')['timestamp'].idxmax()]

    df_filtered.to_csv(file_path, index=False)


if __name__ == "__main__":
    a = requests.get("https://shanghai.anjuke.com/sale/pudong/p50", headers=universal_headers)
    aa = ajk_html_process(a, datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    save_formatted_data("../data/ajk.csv", aa)
