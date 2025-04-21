import sys
import os
import time
import websockets, ssl
import asyncio
# import threading
import argparse
import json
import traceback
import math
from multiprocessing import Process
# from funasr.fileio.datadir_writer import DatadirWriter
import wave
import numpy as np 
import soundfile as sf
import logging
from queue import Queue
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s] %(message)s')


logger = logging.getLogger(__name__)

# 设置logger的等级为DEBUG
logger.setLevel(logging.DEBUG)



def get_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument("--host",
                        type=str,
                        default="localhost",
                        required=False,
                        help="host ip, localhost, 0.0.0.0")
    parser.add_argument("--port",
                        type=int,
                        default=10095,
                        required=False,
                        help="grpc server port")
    
    parser.add_argument("--text",
                        type=str,
                        default=None,
                        help='合成文本')
    
    parser.add_argument('--text_file',
                        type=str,
                        default=None,
                        help='合成文本的文件')
    
    parser.add_argument('--spk',
                        type=str,
                        default='aa',
                        help='指定说话人')
    
    parser.add_argument('--output_dir',
                        type=str,
                        default='',
                        help='输出音频的路径')
    
    parser.add_argument("--thread_num",
                        type=int,
                        default=1,
                        help="thread_num")
    
    parser.add_argument('--prefix',
                        type=str,
                        default='tts_out',
                        help='输出音频的前缀')
    
    
    
    return parser


async def tts_task(data, args):
    text, spk, wav_path = data
    uri = "ws://{}:{}/gpt_sovits/gen_audio".format(args.host, args.port)
    async with websockets.connect(uri) as websocket:
        print("Connected to server")
        await websocket.send(json.dumps({"text": text, "spk": spk}))
        sample_rate = 16000
        audio_data = np.array([], dtype=np.int16)
        while True:
            message = await websocket.recv()
            # print("Received from server:", message)  # 打印接收到的消息

            # 判断是二进制数据还是文本数据
            if isinstance(message, bytes):
                audio_chunk = np.frombuffer(message, dtype=np.int16)
                audio_data = np.concatenate((audio_data, audio_chunk))
            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                    if 'is_finish' in data and data['is_finish'] == 'true':
                            print("Audio recording finished.")
                            sf.write(wav_path, audio_data, sample_rate)
                            break
                except json.JSONDecodeError:
                    print(f"Received text: {message}")
            else:
                raise ValueError("Unsupported message type received.")

        await websocket.close()

def tts_tasks(datas, args):
    for data in datas:
        asyncio.run(tts_task(data, args))


def batch_process(run_job_fun, datas, args, num_job=40):
    
    
    pool = Pool(num_job)
    
    chunk_size = math.ceil(len(datas)/num_job)
    logger.info(f'total {len(datas)} num_job: {num_job} chunk_size: {chunk_size}')
    results = []
    for i in range(num_job):
        chunk_data = datas[i*chunk_size:(i+1)*chunk_size]
        result = pool.apply_async(run_job_fun, args=(chunk_data,args))
        results.append(result)
    pool.close()
    pool.join()
    
    datas = []
    for result in results:
        data = result.get()
        datas.append(data)
    
    return datas
    


if __name__ == '__main__':
    
    parser = get_parse()
    args = parser.parse_args()
    
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir) 
    
    out_dir = args.output_dir
    prefix = args.prefix
    spk = args.spk
    
    audio_idx = 0
    texts = []
    if args.text is not None:
        text = args.text
        wav_path = f'{out_dir}/{prefix}_{audio_idx:03d}.wav'
        texts.append([text,spk, wav_path])
    elif args.text_file is not None:
        with open(args.text_file, 'r') as f:
            for line in f:
                text = line.strip()
                wav_path = f'{out_dir}/{prefix}_{audio_idx:03d}.wav'
                texts.append([text, spk, wav_path])
                audio_idx += 1
    else:
        raise ValueError('text or text_file must be set')
    
    logger.info(f'total {len(texts)} text need to synthesize, thread_num {args.thread_num}')
    
    num_job = args.thread_num 
    
    if num_job == 1:
        result = tts_tasks(texts, args)
        results = [result]
    else:
        results = batch_process(tts_tasks, texts, args, num_job=num_job)