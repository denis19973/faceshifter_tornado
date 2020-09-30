import os
from multiprocessing import Process, Queue

import cv2
from bson.objectid import ObjectId
from tornado import ioloop

from faceshifter.inference import initialize_inference_models, swap_faces
from web.settings import MAX_QUEUE_SIZE, MEDIA_RESULTS_PATH


async def add_waiting_tasks_to_queue(q: Queue, db_table):
    cursor = db_table.find({'result_path': None}).sort('i')
    tasks = await cursor.to_list(length=MAX_QUEUE_SIZE)
    for swap_task in tasks:
        q.put(swap_task['_id'])


class InferenceWorker:
    def __init__(self, q, db_client_callback, db_table_name, device: str = 'cpu'):
        self.q = q
        self.create_db_client_callback = db_client_callback
        self.db_table_name = db_table_name
        self.db_client = None
        self.db_table = None
        # model attributes
        self.device = device
        self.G = None
        self.detector = None
        self.arcface_model = None
        self.p = Process(target=self.do_job, args=(self.q,))
        self.p.start()

    def initialize_inference_models(self):
        if all([self.G, self.detector, self.arcface_model]):
            return
        self.arcface_model, self.detector, self.G = initialize_inference_models(self.device)

    def initialize_database(self):
        self.db_client = self.create_db_client_callback()
        self.db_table = getattr(self.db_client, self.db_table_name)

    def do_job(self, q):
        loop = ioloop.IOLoop()
        loop.run_sync(lambda: self.do_job_async(q))

    async def do_job_async(self, q):
        self.initialize_inference_models()
        self.initialize_database()
        while True:
            swap_task_id = self.q.get()
            assert isinstance(swap_task_id, ObjectId)
            swap_task = await self.db_table.find_one({'_id': swap_task_id})
            if not swap_task:
                self.log('!swap task with _id={swap_task_id} was deleted from database')
                continue
            source_file_raw, target_file_raw = self.open_files(swap_task['source_path'], swap_task['target_path'])
            if source_file_raw is None or target_file_raw is None:
                self.log(
                    f"!{'source' if source_file_raw is None else 'target'} file was not found for {swap_task_id}")
                continue
            result_path = os.path.join(MEDIA_RESULTS_PATH, f'{swap_task_id}.jpg')
            source_face, target_face, _ = self.run_inference_and_save_results(source_file_raw, target_file_raw,
                                                                              result_path)
            write_dict = dict(source_face=source_face, target_face=target_face)
            if not source_face or not target_face:
                write_dict['result_path'] = 'faces_not_found_error'
            else:
                write_dict['result_path'] = result_path
            await self.write_result_to_db(swap_task_id, write_dict)
            self.log(f"Inference done on {swap_task_id}.")

    def run_inference_and_save_results(self, source_file_raw, target_file_raw, result_path: str):
        source_face, target_face, swap_result_image = swap_faces(
            self.arcface_model,
            self.detector,
            self.G,
            self.device,
            source_file_raw,
            target_file_raw,
        )
        if source_face and target_face:
            self.write_image(result_path, swap_result_image)
        return source_face, target_face, swap_result_image

    def open_files(self, source_path, target_path):
        source_file = cv2.imread(source_path)
        target_file = cv2.imread(target_path)
        return source_file, target_file

    def write_image(self, path, img):
        img = cv2.convertScaleAbs(img, alpha=(255.0))
        cv2.imwrite(path, img)

    async def write_result_to_db(self, result_id, update_dict: dict):
        assert isinstance(result_id, ObjectId)
        await self.db_table.update_one({"_id": result_id}, {'$set': update_dict})

    def log(self, msg):
        print(f"[{os.getpid()}]: {msg}")
