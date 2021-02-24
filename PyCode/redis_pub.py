from redis_pubsub import Task
import json


if __name__ == '__main__':
    Task().rcon.publish('task:pubsub:channel', json.dumps({'id': 1, 'value': '你好'}))