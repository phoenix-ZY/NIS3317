from extraction import *
from tqdm import tqdm
import time
import webbrowser

regions = ['pudong', 'minhang', 'songjiang', 'baoshan', 'jiading', 'xuhui', 'qingpu', 'jingan', 'putuo', 'yangpu', 'fengxian', 'huangpu', 'hongkou', 'changning', 'jinshan', 'chongming', 'shanghaizhoubian']
data_path = "../data/ajk.csv"



def build_urls(region_index=None,page_index=0):
    urls = []

    if region_index is None:
        region_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    target_regions = [regions[i] for i in region_index]

    for region in target_regions:
        if page_index==0:
            urls.append(f"https://shanghai.anjuke.com/sale/{region}/")
            page_index+=2
        for i in range(page_index, 51):
            urls.append(f"https://shanghai.anjuke.com/sale/{region}/p{i}/")

    return urls


def main_crawling(urls, interval=5):
    elapsed_time = 0
    with tqdm(urls, desc='Progress') as tbar:
        for idx,url in enumerate(tbar):
            tbar.set_postfix(link=url, last_elapsed_time=elapsed_time)
            tbar.update()

            start_time = datetime.now()
            success = False
            while not success:
                page_rsp = requests.get(url, headers=universal_headers,allow_redirects=False)
                page_data = ajk_html_process(page_rsp, datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
                if page_data is None:
                    print("[Banned! Please Verify in 15 Seconds]")
                    webbrowser.open(url, new=0, autoraise=True)
                    sleep(15)
                else:
                    success = True
            save_formatted_data(data_path, page_data,check_repetition=idx%int(len(urls)/10)==0)
            time.sleep(interval)
            elapsed_time = datetime.now() - start_time
    save_formatted_data(data_path, [], check_repetition=True)


if __name__ == "__main__":
    target_urls = build_urls([16],page_index=0)
    main_crawling(target_urls, interval=0)
