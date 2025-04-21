import os
import websockets
import asyncio
import numpy as np
from scipy.io import wavfile
import json
import soundfile as sf
import time
import logging
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import sys
import uuid
from tqdm import tqdm

def init_file_logger(filename, level = logging.DEBUG, propagate=True):

    logger = logging.getLogger(filename)
    logger.setLevel(level)  # 设置日志级别为DEBUG
    formatter = logging.Formatter('[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s')

    # 创建一个文件处理器，用于写入日志文件
    file_handler = logging.FileHandler(f'logs/{filename}.log')  # 日志文件路径
    file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别
    file_handler.setFormatter(formatter)
    
    # 创建一个流处理器，用于输出到终端
    stream_handler = logging.StreamHandler()  # 输出到终端
    stream_handler.setLevel(level)  # 设置流处理器的日志级别
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 将文件处理器添加到logger中
    logger.addHandler(file_handler)
    logger.propagate = propagate 
    
    
    
    return logger

logger = init_file_logger('batch_request', level = logging.DEBUG, propagate=True)



async def gen_tts(task_data):
    current_pid = os.getpid()
    
    first_time = 0
    other_times = []
    fs = task_data.get('rate', 16000)
    
    task_idx = task_data.get('task_idx', current_pid)    # 并发索引， 进程id
    audio_data = bytearray()
    async with websockets.connect(url, ping_timeout=6000) as ws:
        logger.debug(f'connect success task-{task_idx}')
        await ws.send(json.dumps(task_data))
        st0 = time.time()   # 建立链接开始时间
        last_receive_time = time.time()
        finished = False
        count = 0
        wd_num = 0
        

        st1 = 0 # 第一包收到的开始时间
        full_text = ""
        

        while finished is False:
            message = await ws.recv()
            
            if isinstance(message, bytes):
                receive_interval = time.time() - last_receive_time
                if first_time == 0:
                    first_time = receive_interval
                    st1 = time.time()
                else:
                    other_times.append(receive_interval)
                last_receive_time = time.time()
                
                receive_time = time.time() - st1
                last_audio_time = len(audio_data)/(2*fs)
                
                # 接收音频数据
                audio_data.extend(message)
                
                if len(other_times) > 0 and receive_time > last_audio_time:
                    logger.warning(f'-------------断流 last audio-time {last_audio_time:.1f} receive-time {receive_time:.1f}')
                
            elif isinstance(message, str):
                data = json.loads(message)
                # messageType = data.get('messageType', "")
                # if messageType == 'ttsFinish':
                #     finished = True
                finished = data['finish']
                
                text = ""
                if 'stamp' in data:
                    text = [a['word'] for a in data['stamp']['words']]
                    text = ''.join(text)
                    count += 1
                    wd_num += len(text)
                    full_text += text
                    logger.debug(f'task-{task_idx} receive-{count} : {text}')
                if finished:
                    
                    total_use = time.time() - st0
                    rtf = wd_num/total_use
                    other_avg = sum(other_times)/(len(other_times)+0.00001)
                    audio = np.frombuffer(audio_data, dtype=np.int16)
                    audio_dur = len(audio) / fs
                    # logger.info(f'audio-len {len(audio_data)}, audio-len {len(audio)}')
                    if audio_dur > 0:
                        rtf_audio = total_use/(audio_dur + 0.00001)
                        logger.info(f'task-{task_idx} receive audio finish, total {wd_num} word, total {audio_dur:.1f}s, infer {total_use:.1f}s, rtf-{rtf_audio:.1f} avg {rtf:.1f}w/s, first {first_time:.1f}s, other-avg: {other_avg:.1f} ')
                    else:
                        logger.error(f'task-{task_idx} receive audio error')
            else:
                logger.error(f'receive message error type type {type(message)}')
            

        
        audio = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio) > 0:
            
            out_wav_path = task_data.get('out_wav_path', f'out_wav/{current_pid}.wav')
            ori_text = task_data.get('text', '')
            sf.write(out_wav_path, audio, fs, subtype='PCM_16')
            logger.debug(f'save to {out_wav_path}')
            with open(f'{out_wav_path[:-4]}.txt.ori', 'w') as f:
                f.write(ori_text)
            with open(f'{out_wav_path[:-4]}.txt.tn', 'w') as f:
                f.write(full_text)

async def main(num_batch):
    tasks = []
    for _ in range(num_batch):
        tasks.append(gen_tts(task_data))

    try:
        # 设置最大等待时间为5秒
        # await asyncio.wait_for(asyncio.gather(*tasks), timeout=5000)
        await asyncio.gather(*tasks)
    except asyncio.exceptions.TimeoutError:
        print("Timeout occurred before all tasks completed.")


def sync_gen_tts(task_data):
    # 使用asyncio.run来运行异步函数
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(gen_tts(task_data))
    loop.close()
    return result



def run_gen_tts_in_process_pool(task, num_worker=20):
    results = []

    # 创建固定大小为20的进程池
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        # 提交任务到进程池中执行
        futures = []
        for i in range(num_worker):
            task_data = copy.deepcopy(task)
            task_data['task_idx'] = i+1
            future = executor.submit(sync_gen_tts, task_data)
            futures.append(future)

        # 使用as_completed来获取完成的任务结果
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"An error occurred: {e}")

    return results


def generate_batch_task_datas(text_file, num,  spk_names, out_dir='out_wav/batch_test'):
    
    tts_task = {
        'task_name': 'tts',
        'spk_name': "default_spk",
        'request_id': str(uuid.uuid4()),
        'text': '这是一个测试',
        'language': 'zh',
        'rate': 16000,
        'volume_ratio': 1.0,
        'speed_ratio': 1.0,
        'pitch_ratio': 1.0    
    }
    
    os.makedirs(out_dir, exist_ok=True)
    with open(text_file, 'r') as f:
        texts = [line.strip() for line in f.readlines()]
        texts = [txt for txt in texts if txt != '']
    
    logger.info(f'total {len(texts)} text, run {num} times every text')
    task_datas = []
    
    for spk_name in spk_names:
        job_dir = f'{out_dir}/{spk_name}/'
        os.makedirs(job_dir, exist_ok=True)
        for j, text in enumerate(texts):
            for i in range(num):
                task_data = copy.deepcopy(tts_task)
                task_data['request_id'] = str(uuid.uuid4())
                task_data['text'] = text
                task_data['spk_name'] = spk_name
                task_data['out_wav_path'] = f'{job_dir}/utt_{j}_job_{i}.wav'
                task_datas.append(task_data)
        return task_datas


def batch_process(run_job_fun, task_datas, max_workers=40):
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_job_fun, task_data): task_data for task_data in task_datas}
        
        for future in tqdm(as_completed(futures), total=len(task_datas)):
            task_data = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Task {task_data} generated an exception: {exc}")
                
                

def main2():
    spk_names = ['exp3_g5_s108_baoyi']
    num_times = 100
    num_worker = 20
    sub_dir = 'out_wav/batch_test/exp2_g15_s56/'
    text_file = 'testset2.txt'
    # text_file = 'a'
    task_datas = generate_batch_task_datas(text_file, num_times, spk_names, out_dir=sub_dir)
    batch_process(sync_gen_tts, task_datas, max_workers = num_worker)

port = 8000
url = f'ws://localhost:{port}/ws/'
task_data = {
        'task_name': 'clone',
        'request_id': str(uuid.uuid4()),
        'spk_id': '0ad13431-f3a3-4d0a-865b-320dd2b8868e',
        'text': "心竞争力、共同实现全网高质量发展",
        'language': 'zh',
        'rate': 16000,
        'volume_ratio': 1.0,
        'speed_ratio': 1.0,
        'pitch_ratio': 1.0    
    }

# text = '张家源， 今天天气怎么样，今天天气非常好。'
# text = '2024-07-30东莞中心综合诊断如下：1、综合评价：1）中心综合评价得分是88.17分，全国排名15名，管控较好，请继续保持。2、质量方面：1）中心质量得分是87.25分，目标是85分，全国排名14名，管控较好，请继续保持。2）进港超时库存率是6.06%，目标是5.2%，全国排名42名，管控较差，请继续努力。3）重复进线率是4.72%，目标是4%，全国排名57名，管控较差，请继续努力。4）始发破损率是十万分之30.24，目标是十万分之90，全国是十万分之83.26，全国4名，管控较好，请继续保持。东莞,目的 华北,破损量 51, 东莞,目的 佛山,破损量 45, 东莞,目的 揭阳,破损量 41, 5）倒包不彻底量是23票，请及时关注处理；3、成本方面：1）单票出港运能成本是0.331元，目标是0.368元，全国是0.395元,，全国排名11名，管控较好，请继续保持。东莞,目的中心 广西梧州市直营交换站,出港运能成本 2.43,出港量 1,单票成本 2.428, 东莞,目的中心 拉萨,出港运能成本 1740.75,出港量 1177,单票成本 1.479, 东莞,目的中心 内蒙古呼伦贝尔市交换站,出港运能成本 1000.23,出港量 684,单票成本 1.462, 2）单次操作成本是0.091元，目标是0.097元，全国是0.111元，全国排名17名，管控较好，请继续保持。4、效率方面：1）出港收件平均操作效率是1483件\/h，目标是1320件\/h，全国是1249件\/h，全国排名10名，管控较好，请继续保持。2）进港卸车平均操作效率是2084件\/h，目标是2000件\/h，全国是1440件\/h，全国排名4名，管控较好，请继续保持。3）进港拆包平均操作效率是2065票\/h，目标是2000票\/h，全国是1710票\/h，全国排名3名，管控较好，请继续保持。5、人员方面：1）小时工占比是0（≤5%），固定工出勤率是88.01%(≥86%)，人均效能是1542.57，目标是1420，评估是否需要招聘；2）人员流失率当月累计是15.01%，全国是6.6%，全国排名76名，管控较差，请继续努力。上个月是4.82%，环比趋势变差，需重点关注；中心操作储备,月初在职 12,月末在职 9,入职 0,离职 3,流失率 28.57, 中心操作,月初在职 0,月末在职 0,入职 0,离职 0,流失率 0, 国际仓操作区,月初在职 0,月末在职 0,入职 0,离职 0,流失率 0, 3）稳定性指数是47.5，全国是45.2，管控较好，请继续保持；中心操作储备,在职 9,平均工龄 10.4,稳定性指数 28.9, 6、安全与目视化：1）装车不规范：告警3次，请及时关注；2）月台下站人：告警7次，请及时关注；3）车等货监控：告警200次，请及时关注；4）拉包不码货告警0次，累计告警47次，请注意网点车位码货。	'
text = '2024-07-30东莞中心综合诊断如下：1、综合评价：1）中心综合评价得分是88.17分，全国排名15名，管控较好，请继续保持。2、质量方面：1）中心质量得分是87.25分，目标是85分，全国排名14名，管控较好，请继续保持。2）进港超时库存率是6.06%，目标是5.2%，全国排名42名，管控较差，请继续努力。3）重复进线率是4.72%，目标是4%，全国排名57名，管控较差，请继续努力。4）始发破损率是十万分之30.24，目标是十万分之90，全国是十万分之83.26，全国4名，管控较好，请继续保持。'
# text = '周末我行走在湖边上看着湖面上航行的船只。'
# text = 'CCR读成 试试看, 贵阳-淮安,线路里程() 1776km。'
# text = "未来，圆通只人将做坚信之人，行分享之事，让圆通更智慧，让员工更幸福，让人生更精彩。"
task_data['text'] = text


# asyncio.run(gen_tts(task_data))       # 执行一个单独的任务
# # asyncio.run(main(32))               # 异步执行多个任务
# run_gen_tts_in_process_pool(task_data, num_worker=8)    # 异步改为同步， 在进程池同时执行多个任务

main2()




# if __name__ == '__main__':
#     text = sys.argv[1]

#     task_data['text'] = text
#     asyncio.run(gen_tts(task_data))

