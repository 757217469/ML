import os
import requests
import json
import datetime

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'

    SQLALCHEMY_DATABASE_URI = None
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # 每次请求结束后自动提交数据库修改
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    # 数据库和模型类同步修改
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_POOL_SIZE = 50
    SQLALCHEMY_POOL_TIMEOUT = 3600
    SQLALCHEMY_POOL_RECYCLE = 3600
    # 查询时显示原始sql
    SQLALCHEMY_ECHO = False

    GATEWAY_OPEN = False  # 网关注册标记
    HK_SERVICE_ROOT = ''
    HK_SERVICE_AUTHENTICATION = 'hk-service-authentication'
    HK_SERVICE_VIDEO_DETAIL = '/view/v2/HKBBTVSSK/article/'
    HK_SERVICE_MEDIAINFO = '/media/v1/HKBBTVSSK/media/thirdPartyInfo/'
    # 6.0海客内容索引查询地址
    SEARCH_CONTENT_URL = '/search/every/contentSearch'
    # 6.0海客视频频道查询地址
    HK_COLUMN_CHANNEL_URL = '/column/channels/every/list'
    LOG_DIR = os.path.join(basedir, 'log')
    EXCEPTION_LOG_FILE = os.path.join(LOG_DIR, 'exception.log')
    SERVICE_NAME = 'hk-service-vote'
    SERVICE_PORT = 5004
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_PWD = None
    REDIS_INFO_DB = 1
    REDIS_HISTORY_DB = 2
    REDIS_COUNT_DB = 15
    REDIS_RANGE_DB = 4
    REDIS_RANGE_TABLE = "activity_vote"

    @staticmethod
    def init_app(app):
        pass


class LocalConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
                              'mysql+pymysql://root:mjk5nj123@localhost/hk_vote_2_0'
    # 查询时显示原始sql
    SQLALCHEMY_ECHO = True
    DEBUG = True
    HK_SERVICE_ROOT = 'http://dev.api.haiwainet.com.cn:9000'
    HK_VOTE_CLIENT_URL = "http://dev.haiwainet.local/vote"
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_PWD = None


class DevelopConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
                              'mysql+pymysql://hww_user:HwwSpecial2019@192.168.11.210/hk_vote_2_0'
    # 查询时显示原始sql
    SQLALCHEMY_ECHO = True
    DEBUG = True
    GATEWAY_OPEN = True
    EUREKA_SERVER = 'http://192.168.11.182:8000/eureka'
    HK_SERVICE_ROOT = 'http://192.168.11.200:9000'
    HK_VOTE_SERVICE_ROOT = "http://192.168.11.200:9000"
    HK_VOTE_CLIENT_URL = "http://dev.haiwainet.com.cn/vote"
    REDIS_HOST = "192.168.11.162"
    REDIS_PORT = 6379
    REDIS_PWD = None
    VIDEO_INFO = 'http://192.168.11.200:9000/view/v2/HKBBTVSSK/article/'
    MEDIA_URL = 'http://192.168.11.200:9000/media/v1/HKBBTVSSK/media/info/relationalMediaIds?mediaId='


class TestConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
                              'mysql+pymysql://hww_user:gunk457(Kiev@10.0.17.153/hk_vote_2_0'
    # 查询时显示原始sql
    SQLALCHEMY_ECHO = True
    DEBUG = True
    GATEWAY_OPEN = False
    EUREKA_SERVER = 'http://10.0.17.22:8000/eureka'
    HK_SERVICE_ROOT = "http://10.0.17.39:9000"
    HK_VOTE_SERVICE_ROOT = "http://10.0.17.39:9000"
    HK_VOTE_CLIENT_URL = "http://mk.haiwainet.com.cn/votetest/v2"
    REDIS_HOST = "10.0.16.38"
    REDIS_PORT = 6379
    REDIS_PWD = None
    VIDEO_INFO = 'http://10.0.17.39:9000/view/v2/HKBBTVSSK/article/'
    MEDIA_URL = 'http://10.0.17.39:9000/media/v1/HKBBTVSSK/media/info/relationalMediaIds?mediaId='


class ProductConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
                              'mysql+pymysql://hk_vote:hk_vote_hww123_!@#@10.0.17.220/hk_vote_2_0'
    # 查询时显示原始sql
    SQLALCHEMY_ECHO = True
    DEBUG = True
    GATEWAY_OPEN = True
    EUREKA_SERVER = 'http://10.0.17.22:8000/eureka'
    HK_SERVICE_ROOT = "http://10.0.17.39:9000"
    HK_VOTE_CLIENT_URL = "http://mk.haiwainet.cn/vote/v2"
    REDIS_HOST = "10.0.17.76"
    REDIS_PORT = 6379
    REDIS_PWD = "Tpfw123098"
    VIDEO_INFO = 'http://haiwaivideo.android.haiwainet.cn/view/v2/HKBBTVSSK/article/'
    MEDIA_URL = 'http://haiwaivideo.android.haiwainet.cn/media/v1/HKBBTVSSK/media/info/relationalMediaIds?mediaId='


class APSchedulerJobConfig(object):
    JOBS = [
        {
            'id': 'autosubimit',
            'func': 'app.task:task.save_plus_history',  # 路径：job函数名
            'args': None,
            'trigger': 'interval',
            'seconds': 1
            # 'next_run_time': datetime.datetime.now() + datetime.timedelta(seconds=5)
        }
    ]
    SCHEDULER_API_ENABLED = True
    SQLALCHEMY_ECHO = True


config = {
    'default': ProductConfig,
    'location': LocalConfig,
    'develop': DevelopConfig,
    'testing': TestConfig
}
