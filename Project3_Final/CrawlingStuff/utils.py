import requests
import re


def print_req(response):
    # 请求相关信息
    print("[Request Method]:", response.request.method)
    print("[Request URL]:", response.request.url)

    # 响应状态码
    print("[Status Code]:", response.status_code)

    # 如果状态码是302，获取重定向地址
    if response.status_code == 302:
        redirect_url = response.headers.get('Location')
        print("[Redirect Location]:", redirect_url)

    # 响应时间
    print("[Response Time]:", response.elapsed.total_seconds(), "seconds")

    # 响应头信息
    print("[Response Headers]:")
    for header, value in response.headers.items():
        print(f"    {header}: {value}")

    # 媒体类型
    print("[Media Type]:", response.headers.get('Content-Type', 'N/A'))

    # 响应内容
    print("[Content]:")
    print(response.text)

    # 如果可能的话，解析并打印 JSON 响应
    if 'application/json' in response.headers.get('Content-Type', ''):
        try:
            json_response = response.json()
            print("[JSON Content]:")
            import json
            print(json.dumps(json_response, indent=4))
        except ValueError:
            print("    Unable to parse JSON content")


available_headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cookie": "aQQ_ajkguid=97446FC6-C9B1-4016-B25C-F58CA8231451; sessid=F2584872-6E1C-4D99-B486-F5E11E6B69E5; ajk-appVersion=; ctid=11; fzq_h=458b8b2a9cf8dac47b5cd854c25c6e3b_1717727756646_8979328cf2ae48aeacc3dbe28d498a6c_989271807; id58=CroD4GZicg1pbSEDM8AEAg==; twe=2; fzq_js_anjuke_ershoufang_pc=9cde6ce76889657e15e8c25d6fad856d_1717728586045_25; xxzl_cid=72ee13d99fdd4fcd9b0ebb5d701e3e8c; xxzl_deviceid=cKOMvWy0of3SLHzITdcmnU9u6SsXNBljBH68qMy0a/KQx2YRWcgIsX21SouK23g0; obtain_by=2; isp=true",
    "Host": "shanghai.anjuke.com",
    "Priority": "u=0, i",
    "Sec-Ch-Ua": '"Microsoft Edge";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0"
}

universal_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0"
}


