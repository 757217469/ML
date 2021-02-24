import redis
import json


class Task(object):

    def __init__(self):
        self.rcon = redis.StrictRedis(host='localhost', db=0)
        self.ps = self.rcon.pubsub()
        self.ps.subscribe('task:pubsub:channel')

    def listen_task(self):
        for i in self.ps.listen():
            if i['type'] == 'message':
                try:
                    print("Task get", json.loads(i['data']))
                except:
                    pass


if __name__ == '__main__':
    print('listen task channel')
    Task().listen_task()