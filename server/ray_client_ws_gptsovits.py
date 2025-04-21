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

from conf import SERVER_ROOT_DIR

rds = redis.Redis(host='localhost', port=6379, db=0)

spk_id = 'naijie-5f8cce41-4baf-4abe-acc8-26ee0eb16908'
spk_id = 'naijie-{}'.format(str(uuid.uuid4()))
audio_id = 'naijie-audio-2d718437-40de-4a35-8357-b6b592347b25'
ref_wav_path = f'{SERVER_ROOT_DIR}/clone_wav/naijie_split01.wav'
tts_spk_names = ['chattts-8434864','chattts-41575772','chattts-70630540']
tts_spk_names = ['luoye', 'weiwei', 'baoyi']
tts_spk_names = ['baoyi_g10-s88_split08_au', 'luoying_g10-s88_split04_au_denoise', 'baoyi_g10-s88_split08_au_denoise', 'weiwei_g10-s88_split08_au']
tts_spk_names = ['exp3_g5_s108_baoyi', 'exp3_g5_s108_weiwei']


text = '网点的盈利状况，是由收入提升和成本控制两方面决定的，圆通网点要想赚钱，主要是做好“三升四降”。1、提升服务质量，网点可以通过缩短揽派时长、提高出港交件及时率和签收及时率，努力减少遗失破损和虚假签收等举措来提升服务质量。2、提升客户体验，网点要重点推广好直通总部的快速理赔功能，提升理赔速度，同时要努力降低重复进线率和缩短专属群回复时长，服务质量和客户体验是决定快递价格差异的核心要素。'
# text = '张帅明:工作地址：上海市青浦区华徐公路3029弄18号'
text = '网点的盈利状况，是由收入提升和成本控制两方面决定的'
# text = '我的邮箱是1802140659@qq.com  , 我的邮箱是1802140659@QQ.com '
#text = "2024-08-01郑州中心进港诊断：1、超时库存：进港离场超时库存27243票，当前未解决11695票。379001,网点名称 河南省洛阳市,当日超时未解决票数 461 371069,网点名称 河南省郑州市龙湖,当日超时未解决票数 347371050,网点名称 河南省郑州市中牟县,当日超时未解决票数 308, 2、进港卸车：1）早班待卸车6辆，中班待卸车4辆；实时待卸车5辆，平均等待时长0.59h，等待时长超1小时1辆，待卸87419票，待卸7910件，待卸2846包。AQ951905006954,始发 -,等待时长 1.76, 2）进港卸车平均卸车效率1476.00。3）未来0-1小时到达车3辆，1231包，4463件单件，31061票。1-2小时2辆，916包，3442件单件，27307票。2小时以上49辆，26147包，102273件单件，1009847票。3、进港拆包：1）早班待拆包449个，中班待拆包177个，实时待拆包198个,平均滞留时长0.17h；滞留超2小时以上1包。NW21210156292,建包单位 河北省保定市高碑店市白沟镇,拆包单位 郑州转运中心,滞留时长(h) 3.37, 2）进港分拣操作票量804701，平均操作效率1757.00件\/h，最新操作效率1755.00件\/h。1-DWS-18,最新操作效率 837, 1-DWS-3,最新操作效率 1292, 1-DWS-15,最新操作效率 1304, 3）进港小循环拥堵告警0次，累计告警1次，请注意控制小循环流量。4）进港未建包量712，当日累计未建包量8395，累计占比36.31%，请注意管控，严禁单件进入下包线。4、进港上车：1）进港错装55票，进港错装率0.31%。2）拉包不码货告警0次，累计告警4次，请注意网点车位码货。郑州,最新告警次数 1216,累计告警次数 38, 3）进港车等货累计告警92次。河南省三门峡市义马市,累计告警次数 28,最新告警次数 1, 河南省郑州市郑东新区职教园,累计告警次数 16,最新告警次数 1, 河南省郑州市金水东区,累计告警次数 16,最新告警次数 1,"
port = 7200
url_clone_audio = f'http://localhost:{port}/clone_audio/'
url = f'ws://localhost:{port}/ws/'

tts_task = {
        'task_name': 'tts',
        'spk_name': random.choice(tts_spk_names),
        'request_id': str(uuid.uuid4()),
        'text': text,
        'language': 'zh',
        'rate': 16000,
        'volume_ratio': 1.0,
        'speed_ratio': 1.0,
        'pitch_ratio': 1.0    
    }
clone_task = {
        'task_name': 'clone',
        'request_id': str(uuid.uuid4()),
        'spk_id': spk_id,
        'text': text,
        'language': 'zh',
        'rate': 16000,
        'volume_ratio': 1.0,
        'speed_ratio': 1.0,
        'pitch_ratio': 1.0    
    }

def test_gen_spk_emb():
    ref_speech, sr = librosa.load(ref_wav_path, sr=16000)
    ref_speech = (ref_speech*32768).astype(np.int16)
    audio_base64 = base64.b64encode(ref_speech).decode('utf-8')
    
    task_data = {
        'task_name': 'gen_spk_emb',
        'spk_id': spk_id,
        'speech_data': audio_base64,
        'text': '你好，请问一下， 华兴路怎么走。',
        'language': 'zh'
    }
    
    response = requests.post(url_clone_audio, json=task_data)
    receive_spk_id = response.text
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

async def gen_tts(task_data):
    
    
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
        request_id = str(uuid.uuid4())
        task_data['request_id'] = request_id
        await ws.send(json.dumps(task_data, ensure_ascii=False))
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
                # text = ""
                # if 'stamp' in data:
                #     text = [a['word'] for a in data['stamp']['words']]
                #     text = ''.join(text)
                # print(f'{text}')
                # if finished :
                #     print(f'receive audio finish')
            else:
                print(f'error type type {type(message)}')
            
            # is_break = count==4 #random.choice([True, False, True, ])
            # if is_break:
            #     print(f'-------- client break test ---')
            #     await ws.close()
            #     break
    
        audio = np.frombuffer(audio_data, dtype=np.int16)
        sf.write('out_clone.wav', audio, fs, subtype='PCM_16')
        other_mean = sum(other_times)/len(other_times) if len(other_times) > 0 else -1
        print(f'{request_id} total {len(audio)/fs:.2f}s audio, first_receive {first_time:.2f}s, other-avg {other_mean} write to out_clone.wav')
    

# test_gen_spk_emb()

async def main():
    tasks = []
    for _ in range(10):
        tasks.append(gen_tts(tts_task))
    
    for _ in range(10):
        tasks.append(gen_tts(clone_task))
    
    try:
        # 设置最大等待时间为5秒
        # await asyncio.wait_for(asyncio.gather(*tasks), timeout=5000)
        await asyncio.gather(*tasks)
    except asyncio.exceptions.TimeoutError:
        print("Timeout occurred before all tasks completed.")

# asyncio.run(main())
asyncio.run(gen_tts(tts_task))
