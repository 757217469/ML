# -*- coding: utf-8 -*-
# 开发人员   ：黎工
# 开发时间   ：2019/10/21  14:53
# 文件名称   ：news_post.PY
# 开发工具   ：PyCharm

import re
from urllib.parse import urljoin
from scrapy.selector import Selector
import requests
import json


def get_haike_img_url(images, count=1):
    url = 'http://10.0.17.103:9000/hk-service-file/materials/upload/url'
    headers = {
        'Ccontent-Type':'application/json'
    }
    data = json.dumps(images)
    response = requests.post(url,headers=headers,data=data)
    content = json.loads(response.text)
    print(content)
    if int(content['code']) == 200:
        return [_['imgUrl'] if _['imgUrl'] else _['sourceUrl'] for _ in content['data']]
    else:
        if count < 2:
            count += 1
            return get_haike_img_url(images,count)
        else:
            return images


def parse_content(text, url):
    content = ''
    for t in text:
        t = re.subn(r'<style.*?/style>','', t, flags=re.S)[0]
        t = re.subn(r'<script.*?/script>', '', t, flags=re.S)[0]
        t = re.subn(r'<!--.*?-->', '', t, flags=re.S)[0]
        t = re.subn(r'<(?!table)(?!/table)(?!tr)(?!/tr)(?!td)(?!/td)(?!br)(?!img)(?!p)(?!/p)(?!div)(?!/div)(?!h1)(?!/h1)(?!strong)(?!/strong)[^>]*?>', '', t, flags=re.S)[0]
        if t:
            content += t
    resp = Selector(text=content)
    images = resp.xpath('//img/@src').extract()
    imageUrl = ''
    if images:
        imgs = re.findall('<img.*?>', content, flags=re.S)
        imageUrl = urljoin(url,images[0])
        if imgs:
            images = [urljoin(url, _) for _ in images]
            # haike_url = get_haike_img_url(images)
            haike_url = images
            for _ in range(len(images)):
                if images[_]:
                    # 转成海客url地址
                    content = content.replace(imgs[_], '<img src="%s">' % haike_url[_])
                    # content = re.subn(imgs[_], "<img src='%s'>" % urljoin(url, images[_]), content, flags=re.S)[0]


    content = re.subn(r'<table.*?>', '<table align="center">', content, flags=re.S)[0]
    content = re.subn(r'<tr.*?>', '<tr>', content, flags=re.S)[0]
    content = re.subn(r'<p.*?>', '<p>', content, flags=re.S)[0]
    content = re.subn(r'<div.*?>', '<div>', content, flags=re.S)[0]
    content = re.subn(r'<h1.*?>', '<h1>', content, flags=re.S)[0]
    content = re.subn(r'<strong.*?>', '<strong>', content, flags=re.S)[0]
    content = re.subn(r'<td[^>]*?colspan', 'td style="border:1px solid black;text-align:center;" colspan', content, flags=re.S)[0]
    content = re.subn(r'<td[^>]*?rowspan', 'td style="border:1px solid black;text-align:center;" rowspan', content, flags=re.S)[0]
    content = re.subn(r'<td(?![^>]*colspan)(?![^>]*rowspan).*?', '<td style="border:1px solid black; text-align:center;">', content, flags=re.S)[0]
    content = re.subn(r'\"', '\'', content, flags=re.S)[0]
    right = re.findall('rowspan[^>]*?\'[^>]*?\'([^>]+)', content, flags=re.S)
    right += re.findall('colspan[^>]*?\'[^>]*?\'([^>]+)', content, flags=re.S)
    for _ in right:
        content = re.subn(r'{}'.format(_), '', content, flags=re.S)[0]
    content = re.subn('[\n]|[\r]|[\t]', '', content, flags=re.S)[0]
    if '\xa0\xa0\xa0' in content:
        content_list = content.split('\xa0\xa0\xa0')
        content = ''
        for _ in content_list:
            content += '<p>' + _ + '</p>'

    return (content, imageUrl)
