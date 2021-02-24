# coding=utf-8
import redis
import json


# rdp = redis.StrictRedis(host='localhost',port=6379,db=0,password=None)
# red = redis.ConnectionPool(rdp)
red = redis.Redis(host='localhost', port=6379, db=0)
red = red.pipeline()

items = {_.get('item_name'):[[i.get('topic_name') for i in poS.select(f'select topic_name from "fd_content_topic" where id = {_}')] for _ in _.get('topic')] for _ in items}