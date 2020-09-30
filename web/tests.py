import json
import unittest
import urllib
from multiprocessing import Queue

import cv2
import numpy as np
import torch
from tornado.ioloop import IOLoop
from tornado.testing import AsyncHTTPTestCase

from app import Application as MainApplication, create_db_client
from faceshifter.inference import initialize_inference_models, swap_faces
from web.settings import MAX_QUEUE_SIZE


class CreateUploadTestCase(AsyncHTTPTestCase):
    GET_URL = '/'
    POST_URL = '/upload_image'
    TEST_IMAGE_PATH = "web/uploads/andrew.jpg"

    @classmethod
    def setUpClass(cls):
        queue = Queue(maxsize=MAX_QUEUE_SIZE)
        main_db_client = create_db_client()
        cls.my_app = MainApplication(queue, main_db_client)

    def get_new_ioloop(self):
        return IOLoop.current()

    def test_http_post(self):
        url = self.get_url(self.POST_URL)
        with open(self.TEST_IMAGE_PATH, 'rb') as f:
            source_file = target_file = f.read()
        files_args = dict(source_file=source_file, target_file=target_file)
        response = self.fetch(url, method="POST", body=urllib.parse.urlencode(files_args))  # todo: correctly encode
        body = json.loads(response.body)
        self.assertEqual(400, response.code)
        self.assertIn("msg", body)
        # self.assertIn("id", body)

    def test_http_get(self):
        url = self.get_url(self.GET_URL)
        response = self.fetch(url, method="GET")
        self.assertEqual(response.code, 200)

    def get_app(self):
        return self.my_app


class TestSanityModelCheck(unittest.TestCase):
    DEVICE = "cpu"
    TEST_IMAGE_PATH = "web/uploads/andrew.jpg"

    def setUp(self):
        self.device = torch.device(self.DEVICE)
        self.arcface_model, self.detector, self.G = initialize_inference_models(self.device)

    def test_inference(self):
        test_img = cv2.imread(self.TEST_IMAGE_PATH)
        source_face, target_face, img_result = swap_faces(self.arcface_model, self.detector, self.G, self.device,
                                                          test_img, test_img)
        self.assertTrue(source_face)
        self.assertTrue(target_face)
        self.assertIsInstance(img_result, np.ndarray)
