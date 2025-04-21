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
import torchaudio
import torch
import os
from rediscluster import RedisCluster

# Redis 集群节点配置
startup_nodes = [
    {"host": "10.6.112.58", "port": "6429"},
    {"host": "10.6.112.58", "port": "6430"},
    {"host": "10.6.112.62", "port": "6429"},
    {"host": "10.6.112.62", "port": "6430"},
    {"host": "10.6.112.66", "port": "6429"},
    {"host": "10.6.112.66", "port": "6430"}
]
def test_gen_spk_emb():
    speech, sr = librosa.load(ref_wav_path, sr=None)
    ref_speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(torch.from_numpy(speech)).numpy()

    ref_speech = (ref_speech * 32768).astype(np.int16)
    audio_base64 = base64.b64encode(ref_speech).decode('utf-8')

    task_data = {
        'task_name': 'gen_spk_emb',
        'spk_id': spk_id,
        'speech_data': audio_base64,
        'text': ref_text,
        'language': 'zh'
    }

    key_exists = rds.exists(spk_id)
    if key_exists:
        print(f"spk_id = {spk_id} Key exists.")
        spk_data = rds.get(spk_id)
        print(f'spk_data type: {type(spk_data)}')
        # exit(0)
    else:
        print(f"spk_id = {spk_id} Key does not exist.")

    response = requests.post(url_clone_audio, json=task_data)
    receive_spk_id = response.text
    status_code = response.status_code
    print(f'clone spk_id = {receive_spk_id}, status_code: {status_code}')
    return receive_spk_id


async def gen_tts(task_data):
    request_id = str(uuid.uuid4())
    task_data['request_id'] = request_id
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
        count = 0
        while finished is False:
            message = await ws.recv()
            count += 1
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
                text = ""
                if 'stamp' in data:
                    text = [a['word'] for a in data['stamp']['words']]
                    text = ''.join(text)
                print(f'{text}')
                if finished:
                    print(f'receive audio finish')
            else:
                print(f'error type type {type(message)}')

        audio = np.frombuffer(audio_data, dtype=np.int16)
        sf.write('../resources/out_clone.wav', audio, 22050, subtype='PCM_16')
        other_mean = sum(other_times) / len(other_times)
        print(
            f'total {len(audio) / fs:.2f}s audio, first_receive {first_time:.2f}s, other-avg {other_mean} write to out_clone.wav')

if __name__ == '__main__':
    '''
    - 维维-短信播报2-split1:        你已成功下单，请保持电话畅通，等待业务员取件， 你还可以通过圆通官网，微信公共号等，在线下单。
    - 维维-日常生活1-split1        夏天来喽， 又能吃上西瓜了，我真的太喜欢在空调房里吃西瓜啦，这种感觉。
    - 男声系统-split01            你已成功下单，请保持电话畅通，等待业务员取件。
    - 男声系统-split02            你还可以通过圆通官网，微信公共号等，在线下单，感谢您的支持。
    - 宝怡-系统1-split01            我是您的智能助理小圆，随时为您提供信息查询和简易任务辅助处理，您可以说查派件 、查取件 、查有到未派等等。
    - 宝怡-工作1-split01            未来，圆通只人将做坚信之人，行分享之事，让圆通更智慧，让员工更幸福，让人生更精彩。
    - 宝怡-媒体报道2-split01        你们不仅是圆通国际业务的拓荒者，更要做圆通国际业务的奠基者。
    - 宝怡-媒体报道2-split02        在三月十五号圆通国际精彩计划的见面会上，面对首期入选人员。
    '''

    port = 7200

    try:
        if 'workspace' in os.path.dirname(__file__):
            # 创建 Redis 集群连接
            print("Connected to Redis Cluster")
            rds = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, socket_timeout=5)
        else:
            print("Connected to localhost")
            rds = redis.Redis(host='localhost', port=6379, db=0)
    except Exception as e:
            print(f"Failed to connect to Redis : {e}")
            exit(1)

    url_clone_audio = f'http://localhost:{port}/clone_audio/'
    url = f'ws://127.0.0.1:{port}/ws/'
    
    print(rds.exists('instruct'))
    spk_id = 'instruct'

    text = "你已成功下单，请保持电话畅通，等待业务员取件， 你还可以通过圆通官网，微信公共号等，在线下单。"
    clone_task = {'task_name': 'clone',
                  'request_id': spk_id,
                  'spk_id': spk_id,
                  'text': text,
                  'language': 'zh',
                  'rate': 22050,
                  'volume_ratio': 1.0,
                  'speed_ratio': 1.0,
                  'pitch_ratio': 1.0}

    asyncio.run(gen_tts(clone_task))