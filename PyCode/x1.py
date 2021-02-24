# -*- coding: utf-8 -*-
# 开发人员   ：黎工
# 开发时间   ：2019/10/25  14:01
# 文件名称   ：1.PY
# 开发工具   ：PyCharm

import requests
import datetime
from requests import adapters
from requests import RequestException
from scrapy import Selector
from batch_news.config.user_agent import get_user_agent
import re
from batch_news.config.BloomFilter import *
from concurrent.futures import ThreadPoolExecutor
from batch_news.config.parse_content import parse_content
from batch_news_post import post_news
from batch_news.config.PymysqlPool import PyMySQL

my = PyMySQL()


class News(object):
    def __init__(self):
        self.url = 'http://www.news.cn/'
        self.headers = get_user_agent()

    # 解析网页
    def get_html(self, url, params=None):
        try:
            requests.adapters.DEFAULT_RETRIES = 3
            response = requests.get(url, headers=self.headers, timeout=20, params=params)
            if response:
                response.encoding = response.apparent_encoding
                return response
        except RequestException as e:
            print('1.py解析错误 %s' % url, e)

    # 使用布隆过滤器过滤URL
    def filter_url(self, detail_urls):
        bf = BloomFilter()
        if bf.isContains(detail_urls):
            print('exists!')
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

    # 获取各频道页链接
    def get_url_list(self):
        url_list = []
        # 台湾频道url_tw
        url_tw = 'http://www.news.cn/tw/2015/gd.htm'
        url_list.append(url_tw)
        # 港澳频道url_gangao
        url_gangao = 'http://www.news.cn/gangao/jsxw.htm'
        url_list.append(url_gangao)
        # 体育频道url_sport
        url_sport = 'http://sports.xinhuanet.com/news.htm'
        url_list.append(url_sport)
        # 国际、时政、社会频道url
        for i in range(1, 5):
            url_world = 'http://qc.wa.news.cn/nodeart/list?nid=113667&pgnum={}&cnt=35&tp=1&orderby=1'.format(i)
            url_list.append(url_world)
            url_politics = 'http://qc.wa.news.cn/nodeart/list?nid=113351&pgnum={}&cnt=35&tp=1&orderby=0'.format(i)
            url_list.append(url_politics)
            url_local = 'http://qc.wa.news.cn/nodeart/list?nid=113322&pgnum={}&cnt=35&tp=1&orderby=1'.format(i)
            url_list.append(url_local)
        for url in url_list:
            self.get_url(url)

    # 获取标题链接
    def get_url(self, url, count=0):
        if count < 2:
            try:
                response = self.get_html(url)
                resp = response.text
                title_list = []
                # 国际、时政、社会频道标题列表页 title_urls
                title_urls = re.findall(r'.*?"LinkUrl":"(http://.*?\.htm)".*?', resp, flags=re.S)
                if title_urls:
                    for title_url in title_urls:
                        title_list.append(title_url)
                # 台湾频道标题列表页 title_tw_urls
                title_tw_urls = re.findall(
                    r'<h3><a href="(http://www.xinhuanet.com/tw/.*?)" target="_blank">.*?</a></h3>', resp, flags=re.S)
                if title_tw_urls:
                    for title_tw_url in title_tw_urls:
                        title_list.append(title_tw_url)
                # 港澳频道标题列表页 title_gangao_urls
                title_gangao_urls = re.findall(
                    r'<h3><a href="(http://www.xinhuanet.com/gangao/.*?)" target="_blank">.*?</a></h3>', resp,
                    flags=re.S)
                if title_gangao_urls:
                    for title_gangao_url in title_gangao_urls:
                        title_list.append(title_gangao_url)
                # 体育频道标题列表页 title_sport_urls
                sport_response = Selector(response)
                title_sport_urls = sport_response.xpath(
                    r'//div[@class="conk"]/div[@class="scpd_page_box"]/li/dt/a/@href').extract()
                # print(title_sport_urls)
                if title_sport_urls:
                    for title_sport_url in title_sport_urls:
                        title_list.append(title_sport_url)
                # print(len(title_list), title_list)

                if not title_list:
                    print('1.py标题链接获取有误', url)
                # for test in title_list:
                #     self.get_article(test)

                val_urls = []
                # 使用布隆过滤器去重
                for title_url in title_list:
                    val_url = self.filter_url(title_url)
                    if val_url:
                        val_urls.append(val_url)
                # print(len(val_urls))
                pool = ThreadPoolExecutor(3)
                pool.map(self.get_article, val_urls)
                pool.shutdown(3)

            except Exception as e:
                print(e)
                count += 1
                self.get_url(url, count)
        else:
            print('1.py获取标题链接两次了', url)

    def get_article(self, url, num=0):
        if num < 2:
            try:
                response = self.get_html(url)
                if response:
                    resp = Selector(response)
                    title = resp.xpath(
                        r'//div[@class="header"]//div[@class="h-news"]/div[@class="h-title"]/text() | //div[@class="ffm"]/div[@class="ztnr"]/div[@class="btt"]/h1/text()').extract_first()
                    if title:
                        title = re.subn(r'<[^>]*?>', '', title, flags=re.S)[0].strip()
                        title = re.subn(r'\"', '\'', title)[0]

                    # 获取稿件发布时间
                    reTime = resp.xpath(
                        r'//div[@class="header"]//div[@class="h-news"]/div[@class="h-info"]/span[@class="h-time"]/text() | //div[@class="btt"]/div[@class="gnxx"]/div[@class="sj"]/text()').extract_first()
                    if reTime:
                        re_Time = reTime.strip()
                        s_time = re_Time.split(' ')[0]
                        today = self.get_Yesterday()
                        if s_time in today:
                            releaseTime = s_time + ' ' + re_Time.split(' ')[1] if len(
                                re_Time.split(':')) == 3 else s_time + ' ' + re_Time.split(' ')[1] + ':00'

                            # 获取稿件来源
                            sourceCode = ''
                            source = resp.xpath(
                                r'//div[@class="h-news"]/div[@class="h-info"]/span[2] | //div[@class="btt"]/div[@class="gnxx"]/div[@class="ly"]/text() | //div[@class="header"]//div[@class="h-news"]/div[@class="h-info"]/span[last()]/text()').extract()
                            if source:
                                sources = source[0].strip()
                                if len(source) > 1:
                                    source = re.findall(r'.*?<em id="source"> (.*?)</em>.*?', source[0], flags=re.S)[
                                        0] if '</em>' in source[0] else sources.split('：')[1].strip().replace('</span>',
                                                                                                              '').strip() if len(
                                        sources.split('：')[1]) > 1 else sources.split('：')[0].strip()
                                else:
                                    source = sources.split('：')[1].strip() if '：' in sources else sources
                                # print(source, url)
                            #  sourceCode = my.get_sourceCode(source, self.url) if source in ['新华网', '新华社'] else my.get_sourceCode(source, '')
                            # print(sourceCode, source, url)

                            # 获取文章内容
                            text = resp.xpath(
                                r'//div[@class="main"]//div[@id="p-detail"]/p | //div[@class="zmy"]/div[@class="content"]').extract()
                            result = parse_content(text, url)
                            content, imageUrl = result[0], result[1]

                            # print(url, sourceCode, title, content, releaseTime, imageUrl)
                            if (not title) or (not content) or (not releaseTime):
                                print('1.py获取时间、标题、稿件来源或内容有误', url)
                            else:
                                res = post_news(source, title, content, imageUrl)
                                if '200' in res:
                                    print('post success')
                                else:
                                    print(res, url)

            except Exception as e:
                print('1.py解析详情页有误url:', url, e)
                num += 1
                self.get_article(url, num)
        else:
            print('1.py详情页解析两次了 %s' % url)

    def main(self):
        print('1.py开始爬取：%s' % ' 新华网 http://www.news.cn/')
        self.get_url_list()


if __name__ == "__main__":
    News().main()
