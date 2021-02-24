# -*- coding: utf-8 -*-
# 开发人员   ：黎工
# 开发时间   ：2020/3/30  9:52
# 文件名称   ：batch_news_post.PY
# 开发工具   ：PyCharm
# coding=utf-8
import requests
import json


# 发布接口  用于pgc用户
def post_news(source, title, content, imageUrl):
    pgcId = 147838
    longitude = 116.47636953094482
    latitude = 39.91772408938928
    # token = 'eyJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJTVVQwMV9wZ2NfMTQ2IiwiaWF0IjoxNTgxMzE2OTg1LCJzdWIiOiJ7XCJtZWRpYVR5cGVcIjpcInBnY1wiLFwic291cmNlVHlwZVwiOlwiU1VUMDFcIixcInVzZXJJZFwiOlwiMTQ2XCJ9IiwiZXhwIjozMTYyNjMzOTcwfQ.b-ovmTEYZH8upGxVmfSOeqtRShLwOkcFXLE9o6UX_b8'
    # 发布接口  应用于爬虫
    url = 'http://10.0.17.64:8003/publish/info/batch'
    # 关键词接口
    tagUrl = 'http://10.0.17.67:5002/article/getKeyword'
    tagData = {
        'content': content,
        'size': 20
    }
    tagHeaders = {
        'Content-Type': 'application/json',
    }
    tag = requests.post(tagUrl, headers=tagHeaders, data=json.dumps(tagData)).json().get('data')

    headers = {
        'Content-Type': 'application/json',
        # 'access_token': token,
        # 'autograph': '12345678',
        # 'secret': 'ViEL1JR+FvKr8Vbo/Dq8v69VoigfFGZ5XgREL5ymi6TnWgGOT7Aw9P4my1vEcPWo6a0Vrgt97oif0QBVRMrSlGKQGQA6CWfcPoYARP8nNYjqrLtae/3qNKNRZP4Oam282VDcB5sCZbsPOJgYSCz9xAZLihERBCnel/mE7sE6ciQ='
    }

    sourceCode = achieveSourceCode(source)
    # sourceCode = pgcId

    if imageUrl:
        data = {
            "title": title,  # 稿件标题
            "contentType": 'CT001',  # 稿件类型:
            "content": content,  # 正文
            "priority": 0,  # 权重
            "commentSwitch": 0,
            "columnList": [
                {
                    "columnCode": "1204343405560340481",
                    'columnPath': '1_1204343405560340481_',
                    "sortNum": 0
                }
            ],  # 栏目列表, 推荐列表
            'columnPath': '1_1204343405560340481_',
            "sourceType": "SUT03",
            "productCode": "PRD001",
            "onePublish": "PRD001",
            "author": source,
            "coverType": "CVT01",
            "coverUrlList": [
                imageUrl
            ],
            "tag": tag,
            "sourceCode": sourceCode,  # 稿件来源Id, 可不填
            "pgcId": pgcId,
            'mediaType': 'pgc',
            "longitude": longitude,
            "latitude": latitude,
            "isPublish": True,
            "auditStatus": "AUDIT-01",  # 保存草稿时使用,需要将isPublish 改为False
            'publishStatus': 'PUB-01',
            'coverUrl': [imageUrl]
        }
    else:
        data = {
            "title": title,  # 稿件标题
            "contentType": 'CT001',  # 稿件类型:
            "content": content,  # 正文
            "priority": 0,  # 权重
            "commentSwitch": 0,
            "columnList": [
                {
                    "columnCode": "1204343405560340481",
                    'columnPath': '1_1204343405560340481_',
                    "sortNum": 0
                }
            ],  # 栏目列表, 推荐列表
            'columnPath': '1_1204343405560340481_',
            "sourceType": "SUT03",
            "productCode": "PRD001",
            "onePublish": "PRD001",
            "author": source,
            "coverType": "CVT01",
            # "coverUrlList": [
            #     imageUrl
            # ],
            "tag": tag,
            "sourceCode": sourceCode,  # 稿件来源Id, 可不填
            "pgcId": pgcId,
            'mediaType': 'pgc',
            "longitude": longitude,
            "latitude": latitude,
            "isPublish": True,
            "auditStatus": "AUDIT-01",  # 保存草稿时使用,需要将isPublish 改为False
            'publishStatus': 'PUB-01',
            # 'coverUrl': [imageUrl]
        }
    # print(data)
    # return 1
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # print(response.text)
    return response.json().get('code')


# 获取sourceCode
def achieveSourceCode(source):
    url = 'http://apinews.android.haiwainet.cn/search/every/mediaSearch'
    headers = {'Content-Type': 'application/json'}
    data = {
        "searchType": "pgc",
        "isPartner": 1,
        "nameKeyword": source

    }
    try:
        sourceCode = int(
            requests.post(url, headers=headers, data=json.dumps(data)).json().get('data')[0].get('mediaId'))
    except:
        addUrl = 'http://10.0.17.24:8012/pgc/partners'
        img = 'http://haikenews.static.haiwainet.cn/image/2020/1/16/8f391e8f-ec26-4d22-90a6-3f78adad58dd.png'
        data = {"mediaName": source,
                "logo": img}
        try:
            sourceCode = requests.post(addUrl, headers=headers, data=json.dumps(data)).json().get('data').get('pgcId')
        except:
            sourceCode = 147838
    return sourceCode


if __name__ == '__main__':
    post_news('海外网', '自动抓取测试004', '这是测试内容',
              'http://haikenews.static.haiwainet.cn/image/2020/1/16/8f391e8f-ec26-4d22-90a6-3f78adad58dd.png')