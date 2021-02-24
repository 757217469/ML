# coding=utf-8
import redis
import json
import pickle


red = redis.Redis(host='localhost', port=6379, db=0)
# red.hset('demo1', 'key1', 1)
# red.dump('demo1')
#
# red.hscan('demo1')
# red.hmset()

s = {str(_):{
                "address": "",
                "attention_type": 1,
                "city": "南京市",
                "city_code": "3201",
                "collect_type": 0,
                "comment_count": "",
                "content": {
                    "url": "http://outin-197f68bface111e9acfb00163e06123c.oss-cn-shanghai.aliyuncs.com/b57ee75c0a39431e99635f190aa246c7/a067917757fb493383e031ff249ccaf6-43704b0f10c2d677c5eb6908e456d908-ld.mp4",
                    "video_height": 960,
                    "video_length": 8,
                    "video_width": 540
                },
                "content_type": "",
                "country": "中国",
                "country_code": "0",
                "cover": "https://outin-197f68bface111e9acfb00163e06123c.oss-cn-shanghai.aliyuncs.com/image/default/D23E99E5462E4976A38EE58B094BAD1D-6-2.png",
                "creator": "ZilingWeng",
                "district": "",
                "district_code": "",
                "down_count": "",
                "ext": {},
                "favor_count": "",
                "has_img": "",
                "has_up": False,
                "id": "110000554097",
                "iid": "110000554097",
                "imgs": [
                    "https://outin-197f68bface111e9acfb00163e06123c.oss-cn-shanghai.aliyuncs.com/image/default/D23E99E5462E4976A38EE58B094BAD1D-6-2.png",
                    "http://outin-197f68bface111e9acfb00163e06123c.oss-cn-shanghai.aliyuncs.com/b57ee75c0a39431e99635f190aa246c7/snapshots/92dc93ddc973435a8a8dc0a6a197b75c-00001.jpg",
                    "http://outin-197f68bface111e9acfb00163e06123c.oss-cn-shanghai.aliyuncs.com/b57ee75c0a39431e99635f190aa246c7/snapshots/92dc93ddc973435a8a8dc0a6a197b75c-00001.jpg"
                ],
                "inBlacklist": 0,
                "insert_time": "",
                "item": [],
                "itemInfo": [
                    {
                        "alg_config": None,
                        "article_config": None,
                        "id": 100007,
                        "item_config": None,
                        "item_name": "生活",
                        "item_order": None,
                        "os_key": None,
                        "status": None,
                        "type": None,
                        "video_config": None
                    }
                ],
                "last_update": "Wed, 25 Sep 2019 21:42:18 GMT",
                "latitude": "32.03364",
                "longitude": "118.790197",
                "mediaInfo": {
                    "address": "",
                    "city": "南京市",
                    "city_code": "3201",
                    "country": "中国",
                    "country_code": "0",
                    "cover": "https://outin-197f68bface111e9acfb00163e06123c.oss-cn-shanghai.aliyuncs.com/image/default/D23E99E5462E4976A38EE58B094BAD1D-6-2.png",
                    "describe": "",
                    "district": "",
                    "district_code": "",
                    "icon": "http://tvax2.sinaimg.cn/crop.0.0.640.640.50/45f345a1ly8fulfzw8iwqj20hs0hsdgh.jpg",
                    "latitude": "32.03364",
                    "longitude": "118.790197",
                    "media_id": "127320",
                    "name": "ZilingWeng",
                    "os_key": "",
                    "province": "",
                    "province_code": "32",
                    "source_type": "2"
                },
                "media_id": "127320",
                "mid": "BODgwMDA0NDMyNzc2",
                "os_key": "",
                "province": "",
                "province_code": "32",
                "pub_time": "1566432675653",
                "review_user_id": "",
                "share_count": "",
                "show_count": 256,
                "source_comment_count": "",
                "source_count_lastdate": "",
                "source_id": "",
                "source_type": "",
                "source_up_count": "",
                "source_url": "",
                "source_view_count": "",
                "status": "",
                "summary": "没想到吉娜的中文说的这么好👍\r\r当郎朗下来，只留吉娜一个人在台上的时候，吉娜显得有些害羞，但有可能是主持人让她用中文做自我介绍，她害怕说不好吧。但还是棒棒的！",
                "title": "吉娜·爱丽丝的中文自我介绍",
                "top_type": 0,
                "topic": [
                    1000174
                ],
                "topicInfo": [
                    {
                        "create_time": None,
                        "desc": None,
                        "ext": None,
                        "icon": None,
                        "id": 1000174,
                        "os_key": None,
                        "status": None,
                        "topic_category": None,
                        "topic_name": "娱乐",
                        "type": 1,
                        "uid": None
                    }
                ],
                "unq_id": "",
                "up_count": 0,
                "user_id": "",
                "video_length": "8",
                "view_count": 256,
                "view_img_type": ""
            } for _ in range(0,10000)}
red.hmset('demo1',s)