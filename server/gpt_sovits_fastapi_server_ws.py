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
import copy
import asyncio
from typing import Union, Dict, Any

# import jieba_fast as jieba

from fastapi import FastAPI,   HTTPException, WebSocket, WebSocketDisconnect
# from fastapi import WebSocket, WebSocketDisconnect
# from fastapi.websockets import WebSocket, WebSocketDisconnect, WebSocketState
from fastapi.websockets import  WebSocketState
import argparse




from conf import SERVER_ROOT_DIR, ErrorCode
sys.path.append(SERVER_ROOT_DIR)
from yto_utils import init_file_logger
from gpt_sovits_model_v2 import zoro_gpt_sovits
from speaker_embedding import SpeakerEmbeddingRedis
from tts_front import TTS_Front
from audio_post_process import audio_post_process
from yto_utils import text_rtf_decorator

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s')

logger = init_file_logger('gpt_sovits_server_ray_ws', level=logging.DEBUG, propagate=True)


import numpy as np
from scipy.io.wavfile import write


app = FastAPI()



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
    def __call__(self, text):
        return self.tts_front.process(text)





class GptSovitsWsRay:
    def __init__(self, gpu_id):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        self.model = zoro_gpt_sovits(device=device)
        self.spk_emb_handle = SpeakerEmbeddingsRedisRay()
        self.tts_front_handle = TtsFront()
        self.default_names = self.model.get_spk_names()


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
                audio_save_task = copy.deepcopy(task)
                audio_save_task['speech'] = bytearray()
                audio_save_task['ori_speech'] = bytearray()
                audio_save_task['time_str'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                use_time = {}
                st0 = time.time()
                if task_name == 'clone':
                    spk_id = task['spk_id']
                    spk_data = await  self.spk_emb_handle.get_spk_emb(spk_id)
                    if spk_data is None:
                        logger.error(f'clone task request_id {request_id}, spk_id not exist in redis : {spk_id}')
                        await ws.send_json({'finish': False, 'rt_code':ErrorCode.REDIS_SPKID, 'error': f'no spk_id or data:  {spk_id} '})
                        flag_connect = False
                        break
                    
                    st1 = time.time()
                    texts = self.tts_front_handle(task['text'])
                    use_time['tts_front'] = time.time() - st1
                    audio_save_task['norm_texts'] = texts
                    
                    use_time['infer'] = 0
                    stamp_st = 0
                    first_send = 0
                    total_len = 0
                    num_seg = len(texts)
                    for i, text in enumerate(texts):
                        st2 = time.time()
                        await self.update_connect_state(ws)
                        if ws.client_state == WebSocketState.DISCONNECTED:
                            logger.info(f'--------clone-task detect client {request_id} close task break ws.client_state={ws.client_state} ------')
                            flag_connect = False
                            break
                        try:
                            audio, fs, stamp = self.model.clone_infer(text, task['language'], spk_data, )
                            audio_save_task['ori_fs'] = fs
                            audio_save_task['ori_speech'].extend(audio.tobytes())
                            stamp_json = self.format_stamp(stamp, stamp_st)
                            audio, fs = self.audio_post(audio, fs, task)
                            audio_time = len(audio)/fs
                            total_len += audio.shape[0]
                            stamp_st += audio.shape[0]/fs
                        except UnicodeEncodeError as e:
                            logger.error(f'error: UnicodeEncodeError {e}')
                        except Exception as e:
                            logger.error(f'error: {e}')
                        
                        infer_time = time.time() - st2
                        use_time['infer'] = use_time['infer'] + infer_time
                        audio_save_task['speech'].extend(audio.tobytes())
                        
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
                        logger.info(f'------------- {request_id}  first send {first_send:.2f}  gen {audio_time:.2f} speech use time {infer_time:.2f}  rtf {rtf:.2f},  text {i+1}/{num_seg}: {text}')
                    use_time['all_time'] = time.time() - st0
                    
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
                    self.save_debug_audio(audio_save_task)
                    
                elif task_name == 'tts':
                    use_time = {}
                    st0 = time.time()
                    if 'spk_name' not in task:
                        logger.error(f'tts task, no key spk_name in json')
                        await ws.send_json({'finish': False, 'rt_code':ErrorCode.NO_KEY_IN_TASK, 'error': 'spk_name not in json'})
                        flag_connect = False
                        break
                    spk_name = task['spk_name']
                    
                    st1 = time.time()
                    texts = self.tts_front_handle(task['text'])
                    use_time['tts_front'] = time.time() - st1
                    use_time['infer'] = 0
                    audio_save_task['norm_texts'] = texts
                    first_send = 0
                    stamp_st = 0
                    total_len = 0
                    num_seg = len(texts)
                    for i, text in enumerate(texts):
                        st2 = time.time()
                        await self.update_connect_state(ws)
                        if ws.client_state == WebSocketState.DISCONNECTED:
                            logger.info(f'--------tts-task detect client close task break ws.client_state={ws.client_state} ------')
                            flag_connect = False
                            break
                        
                        try:
                            audio,fs,stamp, = self.model.tts_infer(text, task['language'], spk_name, )
                            audio_save_task['ori_fs'] = fs
                            audio_save_task['ori_speech'].extend(audio.tobytes())
                            stamp_json = self.format_stamp(stamp, stamp_st)
                            audio, fs = self.audio_post(audio, fs, task)
                            audio_time = len(audio)/fs
                            total_len += audio.shape[0]
                            stamp_st += audio.shape[0]/fs
                        except UnicodeEncodeError as e:
                            logger.error(f'error: UnicodeEncodeError {e}')
                        except Exception as e:
                            logger.error(f'error: {e}')
                        
                        infer_time = time.time() - st2
                        use_time['infer'] = use_time['infer'] + infer_time
                        audio_save_task['speech'].extend(audio.tobytes())
                        
                        try:
                            await self.update_connect_state(ws)
                            if ws.client_state == WebSocketState.CONNECTED:
                                await ws.send_json(stamp_json)
                                await ws.send_bytes(audio.tobytes())
                        except WebSocketDisconnect:
                            logger.info(f'--------tts-task detect client close {request_id} task break ws.client_state={ws.client_state} ------')
                            flag_connect = False
                            break
                        except asyncio.CancelledError:
                            logger.info(f'--------clone-task send audio detect client asyncio.CancelledError {request_id} task break ws.client_state={ws.client_state} ------')
                            flag_connect = False
                            break

                        if first_send == 0:
                            first_send = time.time() - st0
                        
                        rtf = infer_time / audio_time
                        logger.info(f'------------- {request_id} first send {first_send:.2f}  gen {audio_time:.2f} speech use time {infer_time:.2f}  rtf {rtf:.2f},  text {i+1}/{num_seg}: {text}')
                    
                    if flag_connect is False:
                        break
                    
                    use_time['all_time'] = time.time() - st0
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
                    
                    self.save_debug_audio(audio_save_task)
                    
                else:
                    logger.error(f'{task_name} is not supported')
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json({'error_info': 'task_name is not supported'})
            
                await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            logger.info(f'-------------- {request_id} except asyncio.CancelledError')
        except WebSocketDisconnect:
            logger.warn(f'---------------- {request_id}  except client disconnect')
            # return
            # await ws.close(code=1012, reason=str('client disconnect'))
        
        except Exception as e:
            logger.error(f"------------- {request_id} except An unexpected error occurred: {e}")
            # if ws.client_state == WebSocketState.CONNECTED:
            #     await ws.send_json({'finish': False, 'rt_code':ErrorCode.OTHER_ERROR, 'error': f'unexpected error: {e}'})
        finally:
            logger.error(f'--------------- {request_id} finally close websocket --------------')



    async def gen_spk_emb(self, task: dict):
        try:
            # 从请求中获取 base64 编码的音频数据
            logger.info(f'gen_spk_emb , text: {task["text"]}')
            spk_id = task.get('spk_id')
            speech_base64 = task.get('speech_data')
            speech_data = base64.b64decode(speech_base64)
            speech_pcm = np.frombuffer(speech_data, dtype=np.int16)
            speech_data = speech_pcm / 32768.0
            speech_data = speech_data.astype(np.float32)
            fs = 16000
            logger.info(f'get spk_id {spk_id}, speech_data {speech_data.shape}, ')
            norm_texts = self.tts_front_handle(task['text'])
            norm_text = "".join(norm_texts)
            spk_data = self.model.generate_spk_emb(speech_data, fs=fs,  text = norm_text, language = task['language'])
            logger.info(f'success gen spk data')
            self.spk_emb_handle.save_spk_emb(spk_id, spk_data)
            logger.info('success save spk data, return {}'.format(spk_id))
            
            sf.write(f'{SERVER_ROOT_DIR}/out_wav/debug_wav/ref_{spk_id}.wav', speech_pcm, 16000, subtype='PCM_16')
            ref_text = task['text']
            with open(f'{SERVER_ROOT_DIR}/out_wav/debug_wav/ref_{spk_id}.txt', 'w') as f:
                f.write(f'{ref_text}\n')
            
            return {"spk_id": spk_id}
        except UnicodeEncodeError as e:
            logger.error(f'error: UnicodeEncodeError {e}')
            
        except Exception as e:
            logger.error(f'error-400 type {type(e)}: {e}')
            raise HTTPException(status_code=400, detail=str(e))


    def audio_post(self, audio, fs, task):
        out_fs = task.get('rate', 16000)
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
    
    def save_debug_audio(self, audio_task: dict):
        task_name = audio_task['task_name']
        request_id = audio_task['request_id']
        if task_name == 'clone':
            spk = audio_task['spk_id']
        elif task_name == 'tts':
            spk = audio_task['spk_name']
        else:
            spk = 'error_spk'
        
        time_str = audio_task['time_str']
        fs = audio_task['rate']
        
        audio = np.frombuffer(audio_task['speech'], dtype=np.int16)
        audio_filename = f'{SERVER_ROOT_DIR}/out_wav/debug_wav/{task_name}_{spk}_{time_str}_{request_id}.wav'
        ori_audio = np.frombuffer(audio_task['ori_speech'], dtype=np.int16)
        ori_audio_filename = f'{SERVER_ROOT_DIR}/out_wav/debug_wav/{task_name}_{spk}_{time_str}_{request_id}_ori.wav'
        text_filename = audio_filename[:-4] + '.txt'
        norm_text_filename = audio_filename[:-4] + '.norm.txt'
        logger.info(f'--------- save audio: {audio_filename}')
        sf.write(audio_filename, audio, fs, subtype='PCM_16')
        sf.write(ori_audio_filename, ori_audio, audio_task['ori_fs'], subtype='PCM_16')
        with open(text_filename, 'w') as f1, open(norm_text_filename, 'w') as f2:
            f1.write('{}\n'.format(audio_task['text']))
            for txt in audio_task['norm_texts']:
                f2.write(f'{txt}\n')





def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7100,
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
    
    model = GptSovitsWsRay(gpu_id)
    
    uvicorn.run(app, host="127.0.0.1", port=port)