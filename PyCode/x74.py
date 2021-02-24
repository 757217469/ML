# -*- coding: utf-8 -*-
# 开发人员   ：黎工
# 开发时间   ：2019/11/15  11:11
# 文件名称   ：64.PY
# 开发工具   ：PyCharm

import requests
import datetime
from requests import adapters
from requests import RequestException
from scrapy import Selector
from user_agent import get_user_agent
import re
from BloomFilter import *
from concurrent.futures import ThreadPoolExecutor
from parse_content import parse_content
# from batch_news.config.PymysqlPool import PyMySQL
from batch_news_post import post_news
from urllib.parse import urljoin
import urllib3


# my = PyMySQL()


class News(object):
    def __init__(self):
        self.url = 'http://www.nen.com.cn/'
        self.headers = get_user_agent()

    # 解析网页
    def get_html(self, url, params=None):
        try:
            requests.adapters.DEFAULT_RETRIES = 3
            urllib3.disable_warnings()
            response = requests.get(url, headers=self.headers, timeout=20, params=params, verify=False)
            if response:
                response.encoding = response.apparent_encoding
                return response
        except RequestException as e:
            print('解析错误 %s' % url, e)

    # 使用布隆过滤器过滤URL
    def filter_url(self, detail_urls):
        bf = BloomFilter()
        if bf.isContains(detail_urls):
            # print('exists!')
            pass
        else:
            # print('not exists!')
            bf.insert(detail_urls)
            return detail_urls

    # 获取今天，昨天时间
    def get_Yesterday(self):
        today = datetime.date.today()
        oneday = datetime.timedelta(days=1)
        yesterday = today - oneday
        return [str(today), str(yesterday)]

    # 获取标题链接
    def get_title_urls(self, url, count=0):
        if count < 2:
            try:
                title_list = []
                # 标题列表页
                response = self.get_html(url)
                if response:
                    resp = Selector(response)
                    title_list = resp.xpath(r'//div[@class="containernews overh"]//ul/li/a/@href').extract()

                # print(len(title_list),url)
                # print(title_list)

                if not title_list:
                    print('74.py标题列表页链接获取有误', url)

                # for test_url in title_list:
                #     self.get_article(test_url)

                val_urls = []
                # 使用布隆过滤器去重
                for title_url in title_list:
                    val_url = self.filter_url(title_url)
                    if val_url:
                        val_urls.append(val_url)
                # print(len(val_urls))
                print('xxx',val_urls)
                pool = ThreadPoolExecutor(3)
                pool.map(self.get_article, val_urls)
                pool.shutdown(3)

            except Exception as e:
                print(e)
                count += 1
                self.get_title_urls(url, count)
        else:
            print('74.py获取标题链接两次了', url)

    def get_article(self, url, num=0):
        if num < 2:
            try:
                response = self.get_html(url)
                if response:
                    resp = Selector(response)
                    title = resp.xpath(r'//div[@class="clearfix w1000_320 text_title fl"]/h1/text()').extract_first()
                    if title:
                        title = re.subn(r'<[^>]*?>', '', title, flags=re.S)[0].strip()
                        title = re.subn(r'\"', '\'', title)[0]
                    # print(title, url)

                    # 获取稿件发布时间
                    get_time = resp.xpath(
                        r'//div[@class="clearfix w1000_320 text_title fl"]/div[@class="box01"]/div[@class="fl"]/text()').extract_first()
                    # print(get_time, url)
                    if get_time:
                        get_time = re.findall(r'.*?(\d+-\d+-\d+ \d+:\d+).*?', str(get_time), flags=re.S)[0]
                        get_time = datetime.datetime.strptime(get_time.strip(), '%Y-%m-%d %H:%M')
                        s_time = str(get_time).split(' ')[0]
                        if s_time in self.get_Yesterday():
                            releaseTime = str(get_time)
                            # print(releaseTime, url)

                            if releaseTime:
                                # 获取稿件来源
                                get_source = resp.xpath(
                                    r'//div[@class="clearfix w1000_320 text_title fl"]/div[@class="box01"]/div[@class="fl"]/text()').extract_first()
                                # print(get_source)
                                sourceCode = ''
                                # print(get_source, source_url)
                                if get_source:
                                    # get_source = re.findall(r'\\n\\t\\t来源：\\r\\n\\r\\n(.*?)\\r\\n\\u3000.*?', str(get_source), flags=re.S)[0]
                                    source = get_source.strip().split('：')[
                                        1] if '：' in get_source else get_source.strip()
                                    source = source.split('-')[0] if '-' in source else source.strip()
                                    source = source.split('/')[0] if '/' in source else source.strip()
                                    # print(source, url)
                                # sourceCode = my.get_sourceCode(source, self.url) if '东北新闻网' in source else my.get_sourceCode(source, '')
                                # print(sourceCode, source, url)

                                # 获取文章内容
                                text = resp.xpath(r'//div[@class="box_con"]/span[@class="cms_block_span"]').extract()
                                result = parse_content(text, url)
                                content, imageUrl = result[0], result[1]

                                # print(url, content, imageUrl)
                                # print(releaseTime, sourceCode, url, title, content,  imageUrl)
                                if (not title) or (not content) or (not releaseTime):
                                    print('74.py获取时间、标题、稿件来源或内容有误', url)

                                else:
                                    res = post_news(source, title, content, imageUrl)
                                    # print(url, source, title, content, imageUrl)
                                    if '200' in res:
                                        print('post success')
                                    else:
                                        print(res, url)

            except Exception as e:
                print('74.py解析详情页有误url:', url, e)
                num += 1
                self.get_article(url, num)
        else:
            print('74.py详情页解析两次了 %s' % url)

    def main(self):
        print('74.py爬取东北新闻网开始：%s' % self.url)
        url_list = ['http://liaoning.nen.com.cn/lnjinriln_new/index.shtml']
        js_url = 'http://liaoning.nen.com.cn/system/count/0008016/000000000000/count_page_list_0008016000000000000.js'
        js_response = self.get_html(js_url)
        if js_response:
            js_resp = js_response.text
            max_page = re.findall(r'.*?var maxpage = (\d+);.*?', js_resp, flags=re.S)
            if max_page:
                max_page = max_page[0]
                for i in range(2):
                    next_url = 'http://liaoning.nen.com.cn/system/count//0008016/000000000000/000/002/c0008016000000000000_{}.shtml'.format(
                        '0' * (9 - len(max_page)) + str((int(max_page) - i)))
                    url_list.append(next_url)
        print(url_list)
        for url in url_list:
            # print(url)
            self.get_title_urls(url)


if __name__ == "__main__":
    # News().main()
    url = 'http://liaoning.nen.com.cn/system/2020/03/31/020998884.shtml'
    News().get_article(url)