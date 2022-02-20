import time
import os
import base64
import requests
import json
import pickle

import cv2

from .exceptions import *
from .meta import *

PLATFORM_SERVICE_URL = os.getenv('PLATFORM_SERVICE_URL', '0.0.0.0:8089')


class SDK:
    def __init__(self, name):
        self.name = name
        self._params_info = None
        self._params = {}
        self._category = None

        self.load_model()

    def load_model(self):
        response = requests.get(f'{PLATFORM_SERVICE_URL}/sdk/model', params={'model_token': self.name})
        if response.status_code != 200:
            print(f'[ERROR]: Model [{self.name}] does not exist')
            raise ModelNotFoundError()
        data = response.json()
        self._category = data['category_id']
        self._params_info = json.loads(data['params'])
        print(f'[INFO][PARAMETERS]: {self._params_info}')
        for k, v in self._params_info.items():
            self._params[k] = v['default']

    def set_params(self, *args, **kwargs):
        if isinstance(args[0], str):
            keys, values = [args[0]], [args[1]]
        if isinstance(args[0], dict):
            keys, values = list(args[0].keys()), list(args[0].values())

        for i, key in enumerate(keys):
            param = self._params_info[key]
            value = values[i]
            if param['type'] == 'slider':
                if not isinstance(value, (int, float)):
                    raise ValueError()
                value = min(max(value, param['interval'][0]), param['interval'][1])
                self._params[key] = value
            if param['type'] == 'int':
                if not isinstance(value, int):
                    raise ValueError()
                self._params[key] = value

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            retval, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode()

        payload = {
            "module_token": self.name,
            "params": self._params,
            "instances": [
                {
                    "filename": "",
                    "roi": [],
                    "file_obj": image_base64
                }
            ]
        }

        st_time = time.time()
        response = requests.post(f'{PLATFORM_SERVICE_URL}/inference', json=payload)
        print("[INFO][INFERENCE][Status response]:", response.status_code)
        if response.status_code == 500:
            raise ConnectionResetError()
        print("[INFO][INFERENCE][Response]:", response.text)
        result = response.json()
        print(f'[INFO][INFERENCE][Time]: {time.time() - st_time}')

        st_time = time.time()
        payload = {'task_token': result['task_token'], 'keep': 'True'}
        for i in range(20):
            response = requests.get(f'{PLATFORM_SERVICE_URL}/inference_result', params=payload)
            print("[INFO][RESULT][Status response]:", response.status_code)
            if response.status_code == 200:
                break
            else:
                print(f"[INFO][RESULT][Try response]: {i+1}/20")
                time.sleep(0.1)
        print("[INFO][RESULT][Response]:", response.text)
        print(f'[INFO][RESULT][Time]: {time.time() - st_time}')
        return response.json()['result']


class SDKDebug:
    def __init__(self, name, script, debug=False):
        self.name = name
        self.script = script
        self.debug = debug
        self._params_info = None
        self._params = {}
        self._category = None
        self.load_model()

    def load_model(self):
        response = requests.get(f'{PLATFORM_SERVICE_URL}/sdk/model', params={'model_token': self.name})
        if response.status_code != 200:
            print(f'model [{self.name}] does not exists')
            return
        data = response.json()
        self._category = data['category_id']
        self._params_info = json.loads(data['params'])
        print(self._params_info)
        for k, v in self._params_info.items():
            self._params[k] = v['default']

    def set_params(self, *args, **kwargs):
        if isinstance(args[0], str):
            keys, values = [args[0]], [args[1]]
        if isinstance(args[0], dict):
            keys, values = list(args[0].keys()), list(args[0].values())

        for i, key in enumerate(keys):
            param = self._params_info[key]
            if param['type'] == 'slider':
                value = min(max(values[i], param['interval'][0]), param['interval'][1])
                self._params[key] = value

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            retval, image_bytes = cv2.imencode('.jpg', image)
        else:
            raise ValueError()

        meta_data = MetaData(params=self._params)
        object_data = ObjectData(input_data={})
        module = self.script()

        object_data.input_data['IMAGE'] = image_bytes
        status, preprocessed_data = module.preprocessing(meta_data, object_data)

        # inference

        preprocessed_data_base64 = base64.b64encode(pickle.dumps(preprocessed_data)).decode()

        payload = {
            "module_token": self.name,
            "debug": self.debug,
            "params": self._params,
            "instances": [
                {
                    "filename": "",
                    "roi": [],
                    "file_obj": preprocessed_data_base64
                }
            ]
        }

        st_time = time.time()
        response = requests.post(f'{PLATFORM_SERVICE_URL}/inference', json=payload)
        print("Status:", response.status_code)
        if response.status_code == 500:
            print("Status:", response.status_code)
        print("Response:", response.text)
        result = response.json()
        print(f'[CREATING TASK]: {time.time() - st_time}')

        st_time = time.time()
        payload = {'task_token': result['task_token'], 'keep': 'True'}
        for i in range(20):
            response = requests.get(f'{PLATFORM_SERVICE_URL}/inference_result/debug', params=payload)
            print("Status:", response.status_code)
            if response.status_code == 200:
                break
            else:
                time.sleep(0.1)
        print(f'[GETING RESULT]: {time.time() - st_time}')
        object_data.infer_result = pickle.loads(base64.b64decode(response.json()['result']))

        status, postprocessed_data = module.postprocessing(meta_data, object_data)

        return postprocessed_data
