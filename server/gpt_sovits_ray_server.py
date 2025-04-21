import sys
import os
import json
import logging
import numpy as np
from datetime import datetime
import torch


import soundfile as sf
import time

from fastapi import FastAPI, Response, UploadFile, File
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from starlette.requests import Request
from starlette.responses import JSONResponse

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle


from conf import GPT_SOVITS_MODEL_DIR, SERVER_ROOT_DIR
sys.path.append(SERVER_ROOT_DIR)
from yto_utils import init_file_logger, download_file
from gpt_sovits_model import zoro_gpt_sovits
from speaker_embedding import SpeakerEmbeddingRedis
from tts_front import TTS_Front

logger = init_file_logger('ray_serve', level=logging.DEBUG, propagate=True)

import io
import numpy as np
from scipy.io.wavfile import write

import redis

# 返回音频文件
def to_audio_response(audio:np.ndarray, fs:int):
    # 将 numpy 数组转换为 PCM 格式的音频数据
    audio_buffer = io.BytesIO()
    write(audio_buffer, fs, audio.astype(np.int16))
    audio_buffer.seek(0)

    # 创建 HTTP 响应，发送二进制数据
    headers = {
        "Content-Type": "audio/wav",
        "Content-Disposition": "attachment; filename=speech.wav"
    }
    return Response(content=audio_buffer.getvalue(), media_type="audio/wav", headers=headers)


# redis speaker embedding
@serve.deployment(num_replicas=1, ray_actor_options={'num_cpus':1, 'num_gpus':0})
class SpeakerEmbeddingsRedisRay(object):
    def __init__(self):
        self.spkEmbs = SpeakerEmbeddingRedis()

    def get_spk_emb(self, spk_id):
        spk_data = self.spkEmbs.get_spk_data(spk_id)
        return spk_data
    
    def save_spk_emb(self, spk_id, spk_data):
        self.spkEmbs.save_spk_data(spk_id, spk_data)

@serve.deployment
class FileDataRedis(object):
    def __init__(self):
        self.rds = redis.Redis(host='localhost', port=6379, db=0)
    
    def get_data(self, data_id):
        return self.rds.get(data_id)

    def save_data(self, data_id, data):
        if self.rds.exists(data_id) is False:
            return None
        self.rds.set(data_id, data)


@serve.deployment(
    num_replicas=2,
    ray_actor_options={'num_cpus':1, 'num_gpus':0.3})
class GptSovitsRay:
    def __init__(self, spk_emb_handle: DeploymentHandle, filedata: DeploymentHandle):
        gpu_id = torch.cuda.current_device()
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        self.model = zoro_gpt_sovits(device=device)
        self.spk_emb_handle = spk_emb_handle
        self.filedata_handle = filedata
        
    async def run_task(self, task:dict):
        task_name = task['task_name']
        logger.info(f'worker: {task_name}-task ')
        if task_name == 'clone':
            spk_data = await self.spk_emb_handle.get_spk_emb.remote(task['spk_id'])
            if spk_data is None:
                return {"error": "no spk_data"}
            audio, fs = self.model.clone_infer(task['text'], task['language'], spk_data, )
            return to_audio_response(audio, fs)
        elif task_name == 'tts':
            spk_name = task['spk_name']
            audio,fs = self.model.tts_infer(task['text'], task['language'], spk_name, )
            return to_audio_response(audio, fs)
        elif task_name == 'gen_spk_emb':
            spk_id = task['spk_id']
            audio_id = task['audio_id']
            speech_data = await self.filedata_handle.get_data.remote(audio_id)
            if speech_data is None:
                logger.info(f'no speech for spk_id: {audio_id}')
                return ""
            speech_data = np.frombuffer(speech_data, dtype=np.int16)/32768
            speech_data = speech_data.astype(np.float32)
            logger.info(f'clone speech: shape {speech_data.shape}')
            spk_data = self.model.generate_spk_emb(speech_data,  text = task['text'], language = task['language'])
            logger.info(f'success gen spk data')
            await self.spk_emb_handle.save_spk_emb.remote(spk_id, spk_data)
            
            logger.info('success save spk data, return {}'.format(spk_id))
            return  spk_id
        elif task_name == 'get_spk_names':
            spk_names = self.model.get_spk_names()
            logger.info(f'get_spk_names res: spk_names: {spk_names}')
            return spk_names
        else:
            logger.error(f'{task_name} is not supported')
            return None
    
    async def __call__(self, http_request:Request):
        task_data = await http_request.json()
        logger.info(f'################# server get task: {task_data}')
        # task_data['done'] = True
        # return JSONResponse(task_data)
        return await self.run_task(task_data)



tts_app = GptSovitsRay.bind(SpeakerEmbeddingsRedisRay.bind(), FileDataRedis.bind())