import os
import random
import string

import bson
import tornado.web
from bson.objectid import ObjectId

from web import settings


class WelcomePageHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Welcome to FaceShifter API.")


class UploadImageHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            source_file = self.request.files['source'][0]
            target_file = self.request.files['target'][0]
        except KeyError:
            self.set_status(400)
            await self.finish(dict(msg="Incorrect request. Provide source_file and target_file in form-data format."))
            return
        source_file_path = self.get_file_path(source_file)
        target_file_path = self.get_file_path(target_file)
        await self.write_file(source_file_path, source_file['body'])
        await self.write_file(target_file_path, target_file['body'])
        insertion_result = await self.insert_swap_task_to_db(source_file_path, target_file_path)
        self.add_task_to_queue(insertion_result)
        self.set_status(201)
        await self.finish(dict(msg="file uploaded", id=str(insertion_result)))

    async def insert_swap_task_to_db(self, source_path, target_path):
        document = {
            'source_path': source_path,
            'target_path': target_path,
            'source_face': None,
            'target_face': None,
            'result_path': None,
        }
        res = await self.settings['db'].swap_task.insert_one(document)
        return res.inserted_id

    async def write_file(self, file_path, content):
        with open(file_path, 'wb+') as output_file:
            output_file.write(content)

    def get_file_path(self, file_obj):
        original_fname = file_obj['filename']
        extension = os.path.splitext(original_fname)[1]
        fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))
        final_filename = fname + extension
        file_path = os.path.join(os.getcwd(), settings.UPLOAD_PATH, final_filename)
        return file_path

    def add_task_to_queue(self, obj_id: ObjectId):
        assert isinstance(obj_id, ObjectId)
        q = self.settings['queue']
        q.put(obj_id)


class SwapResultHandler(tornado.web.RequestHandler):
    async def get(self, swap_id):
        try:
            swap_id = ObjectId(swap_id)
        except bson.errors.InvalidId:
            self.set_status(400)
            await self.finish(dict(msg="Invalid swap id."))
            return
        result = await self.find_swap_obj(swap_id)
        if not result:
            self.set_status(404)
            await self.finish(dict(msg="Swap object not found."))
            return
        if not result['result_path']:
            self.set_status(202)
            await self.finish(dict(msg="Swapping in progress."))
            return
        no_source_face = result['source_face'] is not None and not result['source_face']
        no_target_face = result['target_face'] is not None and not result['target_face']
        if no_source_face or no_target_face:
            self.set_status(400)
            await self.finish(dict(msg=f"Faces not found on: {'source' if no_source_face else ''}"
                                       f"{', target' if no_target_face else ''}"))
        else:
            self.set_status(200)
            result_url = self.get_result_media_url(result['result_path'])
            await self.finish(dict(msg="Swapping done!", result_url=result_url))

    async def find_swap_obj(self, object_id):
        document = await self.settings['db'].swap_task.find_one({"_id": object_id})
        return document

    def get_result_media_url(self, file_path):
        filename = file_path.split('/')[-1]
        return f"{self.request.protocol}://{self.request.host}{settings.MEDIA_RESULTS_URL}{filename}"
