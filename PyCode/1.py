# coding=utf-8
from flask import request, json, jsonify, g, current_app,logging
import redis
import pandas as pd
import json
import requests
import time


# 获取实时视频信息(包含投票信息)
def achieve_video_info(group_id=12345678, iid=1111111111):
    config = current_app.config
    redis_host = str(config.get('REDIS_HOST'))
    redis_port = int(config.get('REDIS_PORT'))
    redis_pwd = config.get('REDIS_PWD')
    redis_info_db = int(config.get('REDIS_INFO_DB'))
    rdp1 = redis.ConnectionPool(host=redis_host, port=redis_port, password=redis_pwd, db=redis_info_db)
    red = redis.StrictRedis(connection_pool=rdp1)

    # 获取redis中的最新数据
    video_info = red.zrangebyscore(str(group_id), iid, iid)
    # 转dict
    video_info = list(map(to_dict, video_info))[0]
    cid = video_info.get('cid')
    url = str(config.get('VIDEO_INFO')) + cid
    try:
        resp = json.loads(requests.get(url, timeout=1).content).get('data').get('item')
    except:
        return {'code': 400, 'msg': '请求详情异常'}
    data = {
        'vote_totals': video_info.get('vote_totals'),
        'rank': video_info.get('group_rank'),
        'differ': video_info.get('differ_vote'),
        'v_url': video_info.get('v_url'),
        'v_title': video_info.get('title'),
        # 'v_content': video_info.get('v_content'),
        'v_content': resp.get('summary'),
        # 'avatar': video_info.get('avatar'),
        'avatar': resp.get('media').get('icon'),
        # 'username': video_info.get('username'),
        'username': resp.get('media').get('name'),
        'group_id': group_id,
        'relation_id': video_info.get('relation_id'),
        'group_number': video_info.get('group_number'),
        'c_number': video_info.get('c_number'),
        # 'thumb': video_info.get('thumb')
        'thumb': resp.get('cover')
    }
    return data


# 投票函数
def vote_plus(group_id=12345678, iid=1111111111, media_id=12345, activity_id=1, relation_id=123456, false_vote=0):
    config = current_app.config
    redis_host = str(config.get('REDIS_HOST'))
    redis_port = int(config.get('REDIS_PORT'))
    redis_pwd = config.get('REDIS_PWD')

    redis_info = int(config.get('REDIS_INFO_DB'))
    redis_count = int(config.get('REDIS_COUNT_DB'))
    redis_history = int(config.get('REDIS_HISTORY_DB'))
    redis_range = int(config.get('REDIS_RANGE_DB'))

    media_url = str(config.get('MEDIA_URL')) + str(media_id)

    rdp1 = redis.ConnectionPool(host=redis_host, port=redis_port, password=redis_pwd, db=redis_info)
    red = redis.StrictRedis(connection_pool=rdp1)
    rdp2 = redis.ConnectionPool(host=redis_host, port=redis_port, password=redis_pwd, db=redis_history)
    red_his = redis.StrictRedis(connection_pool=rdp2)
    rdp3 = redis.ConnectionPool(host=redis_host, port=redis_port, password=redis_pwd, db=redis_count)
    red_count = redis.StrictRedis(connection_pool=rdp3)
    rdp4 = redis.ConnectionPool(host=redis_host, port=redis_port, password=redis_pwd, db=redis_range)
    red_timestamp = redis.StrictRedis(connection_pool=rdp4)

    # 当前media_id关联用户总可投票数 -1
    # 根据当前media_id查询所有关联的media_id,以第一个media作为依据
    # 查询所有media_id接口
    if not false_vote:
        current_app.logger.info('not false vote---media id：'+str(media_id))
        try:
            media_id = requests.get(media_url, timeout=1).json().get('data')[0]
            current_app.logger.info('vote plus:media_id--'+str(media_id))
        except:
            return {
                'code': 3,
                'data': 'media_id不存在'
            }

        # 判断当前用户在当前活动的某段时间内的投票数是否用完
        # 获取当前活动的时间段列表 [(1568817766,1568827766,10),(1568837766,1568847766,10),(1568857766,1578867766,10)]
        timestamps = json.loads(red_timestamp.hget('activity_vote', activity_id))

        # 判断当前时间在哪个时间戳内
        c_time = int(time.time() * 1000)
        try:
            timestamp = [i for i in timestamps if c_time >= i[0] and c_time <= i[1]][0]
        except:
            return {'code': 1, 'data': '活动已过期'}

        # 获取redis用户投票计数表中的当前用户数据
        # current_app.logger.info('get user score:'+str(media_id))
        user = red_count.zrangebyscore(str(activity_id), media_id, media_id)
        # current_app.logger.info('vote plus get user:'+str(user))

        # 判断用户是否投票过
        if user:
            user = json.loads(user[0])
            # current_app.logger.info('vote plus get user json:'+str(user))
            # 需要判断当前时间是否在用户已记录的时间内
            if user.get('lower_time') <= c_time and user.get('upper_time') >= c_time:
                # 需要一个获取当前活动每个周期票数的接口,先定为timestamp[2]
                # 每个周期的票数
                cycle_count = timestamp[2]
                current_app.logger.info('vote plus 周期票数:'+str(cycle_count))
                # 用户当前票数
                count = user.get('count')
                current_app.logger.info('vote plus 用户当前已投票数:'+str(count))
                if count < cycle_count:
                    current_app.logger.info('vote plus 可以投票')
                    count += 1
                    current_app.logger.info('update count score :'+str(media_id))
                    red_count.zremrangebyscore(str(activity_id), media_id, media_id)
                    red_count.zadd(str(activity_id), media_id,
                                   json.dumps({'media_id':media_id, 'lower_time': timestamp[0], 'upper_time': timestamp[1], 'count': count}))
                else:
                    current_app.logger.info('vote plus 用户不能再投票:'+str(media_id))
                    return {
                        'code': 2,
                        'data': '该用户在当前周期内已无票数'
                    }
            else:
                # 周期过期,用户票数重置,删除旧用户记录,添加新记录
                current_app.logger.info('vote plus 重置用户限制:'+str(media_id))
                red_count.zremrangebyscore(str(activity_id), media_id, media_id)
                red_count.zadd(str(activity_id), media_id,
                               json.dumps({'media_id':media_id, 'lower_time': timestamp[0], 'upper_time': timestamp[1], 'count': 1}))
        else:
            red_count.zadd(str(activity_id), media_id,
                           json.dumps({'media_id':media_id, 'lower_time': timestamp[0], 'upper_time': timestamp[1], 'count': 1}))

        # 存入投票历史
        red_his.zadd(str(activity_id), int(time.time() * 1000),
                     json.dumps(
                         {'media_id': media_id, 'relation_id': relation_id, 'time': int(time.time() * 1000) / 1000}))
    all_video = red.zrangebyscore(str(group_id), 0, 999999999999999999)
    df = pd.DataFrame(list(map(to_dict, all_video)))
    # 获取当前视频的rank 先等等 缺条件

    # 票数 +1
    # 当为虚假票数时,为虚假票数增加
    false_vote = int(false_vote)
    if false_vote > 0:
        df.loc[df.iid == iid, 'additive_vote'] = false_vote
    else:
        df.loc[df.iid == iid, 'actual_vote'] += 1
    # 计算总票数
    df['vote_totals'] = df['actual_vote'] + df['additive_vote']
    # 排名
    df['group_rank'] = df.vote_totals.rank(ascending=False, method='min')
    df.group_rank = df.group_rank.apply(lambda x: int(x))
    # 根据排名排序
    df = df.sort_values('group_rank', ascending=True)

    # df = df.sort_values('vote_totals', ascending=False)
    df = df.reset_index(drop=True)
    df.differ_vote = 0

    # 计算和上级排名票数差值
    for i in range(1, len(df)):
        if df.iloc[i].group_rank == 1:
            break
        else:
            # df.loc[i, 'differ_vote'] = df.iloc[i - 1].vote_totals - df.iloc[i].vote_totals
            rank = get_rank(df, i, 1)
            df.loc[i, 'differ_vote'] = int(
                list(rank.get('vote_totals', 0).items())[0][1]) - df.iloc[i].get(
                'vote_totals', 0)

    # 转换dataframe为字典
    datas = list(df.to_dict(orient='index').values())
    # print(datas)
    # 更新redis当前分组数据
    for _ in datas:
        red.zremrangebyscore(str(group_id), _['iid'], _['iid'])
        red.zadd(str(group_id), _['iid'], json.dumps(_))
    # 返回当前更新的数据
    # 当前视频信息转字典
    # list(df.loc[df['iid'] == iid].to_dict(orient='index').values())[0]

    # 获取返回值data
    data = {
        'differ': list(df.loc[df['iid'] == iid].get('differ_vote', 0).items())[0][1],
        'rank': list(df.loc[df['iid'] == iid].get('group_rank', 0).items())[0][1],
        'vote_totals': list(df.loc[df['iid'] == iid].get('vote_totals', 0).items())[0][1],
    }
    res = {
        'code': 0,
        'data': data
    }
    return res


def to_dict(data):
    return json.loads(data.decode())


def get_rank(df, i, step=1):
    if df.iloc[i].group_rank == 1:
        return 0
    rank = df.loc[df['group_rank'] == (df.loc[i, 'group_rank'] - step)]
    if rank.empty:
        rank = get_rank(df, i, step + 1)
    return rank


if __name__ == '__main__':
    print(vote_plus(group_id=544839000000659, iid=110000538786, media_id=112583, activity_id=546549330000282,
                    relation_id=543840910000153, false_vote=0))
    # print(achieve_video_info())
