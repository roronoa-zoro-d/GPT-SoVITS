import websockets
import asyncio
import numpy as np
from scipy.io import wavfile
import redis 
import librosa
import requests
import json
import soundfile as sf
import base64
import uuid
import random
import time
import sys
from pypinyin import pinyin, Style
import logging

from yto_utils import init_file_logger

logger = init_file_logger('run_batch', level=logging.DEBUG, propagate=True)


from conf import SERVER_ROOT_DIR

sys.path.append('/home/zhangjiayuan/bin/')
from zoro_utils import read_text, read_wav_scp

rds = redis.Redis(host='localhost', port=6379, db=0)


port = 8000
url_clone_audio = f'http://localhost:{port}/clone_audio'
url = f'ws://localhost:{port}/ws/'




def to_pinyin(label_ori):
    pinyin_text = pinyin(label_ori, style=Style.NORMAL, heteronym=False)
    label = ''.join(''.join([item for item in pinyin_word]) for pinyin_word in pinyin_text)
    return label


def gen_spk_emb(spk_id, ref_wav_path, clone_text):
    ref_speech, sr = librosa.load(ref_wav_path, sr=16000)
    ref_speech = (ref_speech*32768).astype(np.int16)
    audio_base64 = base64.b64encode(ref_speech).decode('utf-8')
    
    task_data = {
        'task_name': 'gen_spk_emb',
        'spk_id': spk_id,
        'speech_data': audio_base64,
        'text': clone_text,
        'language': 'zh'
    }
    
    response = requests.post(url_clone_audio, json=task_data)
    data = eval(response.text)
    receive_spk_id = data['spk_id']
    status_code = response.status_code
    print(f'clone spk_id = {receive_spk_id}, status_code: {status_code}')
    
    key_exists = rds.exists(spk_id)
    if key_exists:
        print("spk_id Key exists.")
        spk_data = rds.get(spk_id)
        print(f'spk_data type: {type(spk_data)}')
    else:
        print("spk_id Key does not exist.")
        exit(0)
    
    return receive_spk_id



async def run_task(task_data, out_wav):
    
    key_exists = rds.exists(spk_id)
    if key_exists:
        print("spk_id Key exists.")
    else:
        print("spk_id Key does not exist.")
        await asyncio.sleep(1)
    
    first_time = 0
    other_times = []
    
    fs = task_data['rate']
    audio_data = bytearray()
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(task_data))
        st = time.time()
        finished = False
        while finished is False:
            message = await ws.recv()
            
            if isinstance(message, bytes):
                if first_time == 0:
                    first_time = time.time() - st
                else:
                    other_times.append(time.time() - st)
                # 接收音频数据
                audio_data.extend(message)
                st = time.time()
            elif isinstance(message, str):
                # 解析 JSON 结束标志
                data = json.loads(message)
                finished = data.get('finish', False)
                logger.info(f'{data}')
                if finished :
                    logger.info(f'receive audio finish')
            else:
                logger.error(f'error type type {type(message)}')
    
        audio = np.frombuffer(audio_data, dtype=np.int16)
        sf.write(out_wav, audio, fs, subtype='PCM_16')
        other_mean = sum(other_times)/len(other_times)
        logger.info(f'total {len(audio)/fs:.2f}s audio, first_receive {first_time:.2f}s, other-avg {other_mean} write to {out_wav}')
    

async def batch_run_task(task_datas):
    tasks = []
    for task_data, out_wav in task_datas:
        tasks.append(run_task(task_data, out_wav))
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    
    text_file = sys.argv[1]
    wav_scp_file = sys.argv[2]
    tts_text_file = sys.argv[3]
    
    out_dir = 'out_wav/'
    
    utts, utt2text = read_text(text_file)
    utts, utt2wav = read_wav_scp(wav_scp_file)
    
    with open(tts_text_file, 'r') as f:
        tts_texts = [line.strip() for line in f.readlines()]
    
    task_datas = []
    for utt in utts:
        clone_text = utt2text[utt]
        clone_wav = utt2wav[utt]
        spk_id = str(uuid.uuid4())
        
        clone_spk_id = gen_spk_emb(spk_id, clone_wav, clone_text)
        if clone_spk_id != spk_id:
            logger.error(f'spk_id: {spk_id} not equal to clone_spk_id: {clone_spk_id}')
            exit(0)
        logger.info(f'gen spk_emb utt: {utt} spk_id {spk_id}')
        
        for i, text in enumerate(tts_texts):
            request_id = str(uuid.uuid4())
            out_wav = f'{out_dir}/{utt}_clone_{i}.wav'
            clone_task = {
                    'task_name': 'clone',
                    'request_id': request_id,
                    'spk_id': spk_id,
                    'text': text,
                    'language': 'zh',
                    'rate': 16000,
                    'volume_ratio': 1.0,
                    'speed_ratio': 1.0,
                    'pitch_ratio': 1.0    
                }
            task_datas.append([clone_task, out_wav])

    asyncio.run(batch_run_task(task_datas))

    
            