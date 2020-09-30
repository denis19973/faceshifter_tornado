import multiprocessing
from multiprocessing import Queue

import motor
import tornado.web
from tornado import httpserver
from tornado import ioloop
from tornado.options import define, options

from web.handlers import SwapResultHandler, UploadImageHandler, WelcomePageHandler
from web.inference_worker import InferenceWorker, add_waiting_tasks_to_queue
from web.settings import DATABASE, INFERENCE_WORKERS_COUNT, MAX_QUEUE_SIZE, MEDIA_RESULTS_PATH, MEDIA_RESULTS_URL, \
    SERVER_PORT

define("port", default=SERVER_PORT, type=int)


class Application(tornado.web.Application):
    def __init__(self, db, queue):
        handlers = [
            (r"/", WelcomePageHandler),
            (r"/upload_image", UploadImageHandler),
            (r"/result/([a-zA-Z\-0-9]+)\/?", SwapResultHandler),
            (rf"{MEDIA_RESULTS_URL}(.*)", tornado.web.StaticFileHandler, {"path": MEDIA_RESULTS_PATH},),
        ]
        tornado.web.Application.__init__(self, handlers, db=db, queue=queue)


def create_db_client():
    db_client = motor.motor_tornado.MotorClient(DATABASE['ip'], DATABASE['port'])
    return db_client[DATABASE['name']]


def main():
    multiprocessing.set_start_method('spawn', force=True)
    queue = Queue(maxsize=MAX_QUEUE_SIZE)
    main_db_client = create_db_client()
    ioloop.IOLoop.current().run_sync(lambda: add_waiting_tasks_to_queue(queue, main_db_client.swap_task))
    for _ in range(INFERENCE_WORKERS_COUNT):
        InferenceWorker(queue, create_db_client, 'swap_task')
    print(f'{INFERENCE_WORKERS_COUNT} inference workers started.')
    http_server = httpserver.HTTPServer(Application(main_db_client, queue))
    http_server.listen(options.port)
    ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
