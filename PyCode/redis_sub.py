from redis_pubsub import Task

if __name__ == '__main__':
    print('listen task channel')
    Task().listen_task()