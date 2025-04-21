import sys
import os
import json
import logging
import numpy as np
from datetime import datetime
import torch
import base64
import asyncio
import soundfile as sf
import time
import asyncio
from typing import Union, Dict, Any
import traceback
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from fastapi import FastAPI,   HTTPException, WebSocket, WebSocketDisconnect
from fastapi.websockets import  WebSocketState
import soundfile as sf

import argparse


from conf import COSYVOICE_ROOT_DIR,MATCHA_ROOT_DIR,COSYVOICE_MODEL_DIR, ErrorCode,COSYVOICE_ROOT_DIR_V1
sys.path.append(COSYVOICE_ROOT_DIR)
sys.path.append(MATCHA_ROOT_DIR)
sys.path.append(COSYVOICE_ROOT_DIR_V1)
from yto_utils import init_file_logger
from cosyvoice_infer_v1 import CosyVoiceInfer
from speaker_embedding import SpeakerEmbeddingRedis
# from tts_front import TTS_Front
from tts_front_cosyvoice import TTS_Front
from audio_post_process import audio_post_process
from yto_utils import text_rtf_decorator

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s')

logger = init_file_logger('cosyvoice_server_ws', level=logging.DEBUG, propagate=True)


import numpy as np
from scipy.io.wavfile import write


app = FastAPI()


# redis speaker embedding
class SpeakerEmbeddingsRedisRay(object):
    def __init__(self):
        self.spkEmbs = SpeakerEmbeddingRedis()

    def get_spk_emb(self, spk_id):
        spk_data = self.spkEmbs.get_spk_data(spk_id)
        return spk_data
    
    def save_spk_emb(self, spk_id, spk_data):
        self.spkEmbs.save_spk_data(spk_id, spk_data)



class TtsFront(object):
    def __init__(self, ):
        self.tts_front = TTS_Front()
        
    @text_rtf_decorator(1)
    def __call__(self, text,split=True):
        return self.tts_front.process(text,split)



class CosyVoiceWsRay:
    def __init__(self,gpu_id=0):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        # self.model = CosyVoice(COSYVOICE_MODEL_DIR,device=device,fp16=True,load_jit=True)
        self.model = CosyVoiceInfer(COSYVOICE_MODEL_DIR, use_jit=True, fp16=True, device=device)  # 加载模型

        self.spk_emb_handle = SpeakerEmbeddingsRedisRay()
        self.tts_front_handle = TtsFront()
        print('服务初始化成功')
        # self.default_names = self.model.get_spk_names()


    async def update_connect_state(self, ws: WebSocket):
        # await asyncio.sleep(0.1)
        # return
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await asyncio.sleep(0.1)
                await asyncio.wait_for(ws.receive_text(), 0.01)
        except WebSocketDisconnect:
            cur_state = ws.client_state
            logger.info(f'$$$$$$$$$$$$$ update_connect_state client disconnect {cur_state}')
        except asyncio.TimeoutError:
            cur_state = ws.client_state
        except asyncio.CancelledError:
            logger.info(f'$$$$$$$$$$$$$ update_connect_state task cancelled')


    async def run_task(self, ws: WebSocket):
        await ws.accept()
        request_id = ""
        try:
            flag_connect = True
            while flag_connect:

                task = await ws.receive_json()
                task_name = task['task_name']
                request_id = task['request_id']
                logger.info(f'----------------------------------------------------------------------------------------------------------------------')
                logger.info(f'############ run_task {request_id} :  {task}')
                
                use_time = {}
                st0 = time.time()
                if 'spk_id' in task:
                    spk_id = task['spk_id']
                else:
                    spk_id = task['spk_name']
                spk_data = self.spk_emb_handle.get_spk_emb(spk_id)
                if spk_data is None:
                    logger.error(f'clone task request_id {request_id}, spk_id not exist in redis : {spk_id}')
                    await ws.send_json({'finish': False, 'rt_code':ErrorCode.REDIS_SPKID, 'error': f'no spk_id or data:  {spk_id} '})
                    flag_connect = False
                    break
                
                st1 = time.time()
                texts = self.tts_front_handle(task['text'])

                use_time['tts_front'] = time.time() - st1
                use_time['infer'] = 0
                stamp_st = 0
                first_send = 0
                total_len = 0
                num_seg = len(texts)
                print(f"input_text = {task['text']}")
                audios = []
                for i, text in enumerate(texts):
                    print(f'{os.getpid()}      cur_text_index = {i+1}, text = {text}')
                    st2 = time.time()
                    await self.update_connect_state(ws)
                    if ws.client_state == WebSocketState.DISCONNECTED:
                        logger.info(f'--------clone-task detect client {request_id} close task break ws.client_state={ws.client_state} ------')
                        flag_connect = False
                        break
                    
                    if 'instruct' in spk_id:
                        logger.info(f'--------  instruct  infer  ------')
                        audio, fs, stamp = self.model.inference_sft(text, spk_id='weiwei')
                    else:
                        logger.info(f'--------  clone  infer  ------')
                        audio, fs, stamp = self.model.clone_infer(text, spk_data)

                    audios.append(audio)
                    stamp_json = self.format_stamp(stamp, stamp_st)
                    audio, fs = self.audio_post(audio, fs, task)
                    audio_time = len(audio)/fs
                    total_len += audio.shape[0]
                    stamp_st += audio.shape[0]/fs
                    
                    infer_time = time.time() - st2
                    use_time['infer'] = use_time['infer'] + infer_time
                    
                    try:
                        await self.update_connect_state(ws)
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json(stamp_json)
                            await ws.send_bytes(audio.tobytes())
                    except WebSocketDisconnect:
                        logger.info(f'--------clone-task send audio detect client close {request_id} task break ws.client_state={ws.client_state} ------')
                        flag_connect = False
                        break
                    except asyncio.CancelledError:
                        logger.info(f'--------clone-task send audio detect client asyncio.CancelledError {request_id} task break ws.client_state={ws.client_state} ------')
                        flag_connect = False
                        break
                    
                    if first_send == 0:
                        first_send = time.time() - st0
                    rtf = infer_time / audio_time
                    logger.info(f'-------------first send {first_send:.2f}  gen {audio_time:.2f} speech use time {infer_time:.2f}  rtf {rtf:.2f},  text {i+1}/{num_seg}: {text}')
                use_time['all_time'] = time.time() - st0
                audios = np.concatenate(audios)
                # sf.write(f"../resources/{task['request_id']}.wav", audio_data, 22050, subtype='PCM_16')

                if flag_connect is False:
                    break
                
                final_json = {'finish': True, "speech_len":total_len, 'use_time':use_time}
                
                try:
                    await self.update_connect_state(ws)
                    if ws.client_state == WebSocketState.CONNECTED:
                        logger.info(f'-------------final send {final_json}')
                        await ws.send_json(final_json)
                except WebSocketDisconnect:
                    logger.info(f'--------detect client close task break ws.client_state={ws.client_state} ------')
                    flag_connect = False
                    break
                except asyncio.CancelledError:
                        logger.info(f'--------clone-task send audio detect client asyncio.CancelledError {request_id} task break ws.client_state={ws.client_state} ------')
                        flag_connect = False
                        break
                await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            logger.info(f'-------------- except asyncio.CancelledError')
        except WebSocketDisconnect:
            logger.warn('----------------  except client disconnect')
            # return
            # await ws.close(code=1012, reason=str('client disconnect'))
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            formatted_tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            logger.error(f"------------- except An unexpected error occurred: {formatted_tb}")
            # if ws.client_state == WebSocketState.CONNECTED:
            #     await ws.send_json({'finish': False, 'rt_code':ErrorCode.OTHER_ERROR, 'error': f'unexpected error: {e}'})
        finally:
            logger.error(f'--------------- finally close websocket --------------')


    async def gen_spk_emb(self, task: dict):
        try:
            # 从请求中获取 base64 编码的音频数据
            logger.info(f'gen_spk_emb , text: {task["text"]}')
            spk_id = task.get('spk_id')
            speech_base64 = task.get('speech_data')
            prompt_speech = base64.b64decode(speech_base64)
            prompt_speech = np.frombuffer(prompt_speech, dtype=np.int16)/32768.0
            prompt_speech = prompt_speech.astype(np.float32)
            logger.info(f'get spk_id {spk_id}, speech_data {prompt_speech.shape}, ')

            # 参考文本
            prompt_text = await self.tts_front_handle(task['text'],split=False)
            print(f'提示输入    text = {prompt_text}  speech = {prompt_speech.shape}')

            # 音色克隆
            spk_data = self.model.generate_spk_emb(prompt_text,prompt_speech)
            logger.info(f'success gen spk data')
            self.spk_emb_handle.save_spk_emb(spk_id, spk_data)
            logger.info('success save spk data, return {}'.format(spk_id))
            return {"spk_id": spk_id}
        
        except UnicodeEncodeError as e:
            logger.error(f'error: UnicodeEncodeError {e}')

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            print(tb)
            raise HTTPException(status_code=400, detail=str(e))

    def audio_post(self, audio, fs, task):
        out_fs = task.get('rate', 22050)
        volume_ratio = task.get('volume_ratio', 1.0)
        speed_ratio = task.get('speed_ratio', 1.0)
        pitch_ratio = task.get('pitch_ratio', 1.0)
        logger.info(f'audio_post_process, fs={fs}, out_fs={out_fs}, volume_ratio={volume_ratio}, speed_ratio={speed_ratio}, pitch_ratio={pitch_ratio}')
        audio = audio_post_process(audio, fs, out_fs, volume_ratio, speed_ratio, pitch_ratio)
        return audio, out_fs
    
    def format_stamp(self, data, stamp_st):
        words = []
        for wd, st, ed in data:
            st0 = round(stamp_st+st, 2)
            ed0 = round(stamp_st+ed, 2)
            words.append({"end_time":ed0,"unit_type":"text","start_time": st0,"word":wd})
        stamp_json = {"words": words}
        res = {"stamp": stamp_json, "finish":False}
        return res



def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7200,
        help="serve port",
    )
    
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="gpu-id",
    )
    
    return parser


@app.websocket("/ws/")
async def tts_infer(ws: WebSocket):
    await model.run_task(ws)
    
@app.post('/clone_audio/')
async def gen_spk_emb(task: dict):
    await model.gen_spk_emb(task)


if __name__ == "__main__":
    import uvicorn
    
    parser = get_parser()
    args = parser.parse_args()
    gpu_id = args.gpu_id
    port = args.port
    
    model = CosyVoiceWsRay(gpu_id)
    
    uvicorn.run(app, host="127.0.0.1", port=port)
