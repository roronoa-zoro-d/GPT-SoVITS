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
from concurrent.futures import ProcessPoolExecutor
import copy
import sys
import random

def init_file_logger(filename, level = logging.DEBUG, propagate=True):

    logger = logging.getLogger(filename)
    logger.setLevel(level)  # 设置日志级别为DEBUG
    formatter = logging.Formatter('[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s')

    # 创建一个文件处理器，用于写入日志文件
    file_handler = logging.FileHandler(f'logs/{filename}.log')  # 日志文件路径
    file_handler.setLevel(level)  # 设置文件处理器的日志级别
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
    
    
    first_time = 0
    other_times = []
    fs = 22050
    
    task_idx = task_data.get('task_idx', -1)
    audio_data = bytearray()
    time.sleep(random.randint (1,5))
    async with websockets.connect(url, ping_timeout=6000) as ws:
        ori_text = task_data.get('text', '')
        logger.info(f'connect success task-{task_idx}, {ori_text}')
        await ws.send(json.dumps(task_data))
        st0 = time.time()
        st = time.time()
        finished = False
        count = 0
        wd_num = 0
        cur_audio_time = 0
        receive_time = 0
        last_receive_time = -1
        while finished is False:
            message = await ws.recv()
            
            if isinstance(message, bytes):
                if first_time == 0:
                    first_time = time.time() - st
                else:
                    other_times.append(time.time() - st)
                # 接收音频数据
                audio_data.extend(message)
                cur_audio_time = len(audio_data)/(2*fs)
                receive_interval = time.time() - last_receive_time
                if last_receive_time != -1 and receive_interval > cur_audio_time:
                    logger.error(f'-------------断流 cur audio-time {cur_audio_time:.1f} receive-time {receive_interval:.1f}')
                last_receive_time = time.time()
                st = time.time()
            elif isinstance(message, str):
                # 解析 JSON 结束标志
                data = json.loads(message)
                messageType = data.get('messageType', "")
                if messageType == 'ttsFinish':
                    finished = True
                # print(data)
                
                text = ""
                if 'words' in data:
                    text = [a['word'] for a in data['words']]
                    text = ''.join(text)
                    count += 1
                    wd_num += len(text)
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
                        logger.warning(f'task-{task_idx} receive audio error, message: {message}')
            else:
                print(f'error type type {type(message)}')
            

        current_pid = os.getpid()
        audio = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio) > 0:
            sf.write(f'out_wav/{current_pid}.wav', audio, fs, subtype='PCM_16')
            logger.info(f'save to out_wav/{current_pid}.wav')

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
            #task_data['text'] = f'测试任务{i},' + task_data['text']
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


port = 9093
url = f'ws://10.130.118.118:{port}/tts'
task_data = {
	"source": "JSC_APP",
	"appSecret": "sCnuN1dBjiof1hEI5g0zv",
	"modelCode": "Test_app_h56",
	"userCode": "02191859",
	"type": "TTS",
	"token": "1986dc8072b460a75942e484cc08f1c1",
	"appkey": "",
	"channel": 5,
	"fontName": "cosyvoice-weiwei",
	"speedLevel": 1,
	"volume": 1,
	"text": "心竞争力、共同实现全网高质量发展"
}
# task_data = {
# 	"source": "JSC_APP",
# 	"appSecret": "sCnuN1dBjiof1hEI5g0zv",
# 	"modelCode": "Test_app_h56",
# 	"userCode": "02191859",
# 	"type": "TTS",
# 	"token": "9a5447aa07569108e0e898663e684a9f",
# 	"appkey": "",
# 	"channel": 5,
# 	"fontName": "weiwei",
# 	"speedLevel": 1,
# 	"volume": 1,
# 	"text": "心竞争力、共同实现全网高质量发展"
# }

# text = '张家源， 今天天气怎么样，今天天气非常好。'
# text = '2024-07-30东莞中心综合诊断如下：1、综合评价：1）中心综合评价得分是88.17分，全国排名15名，管控较好，请继续保持。2、质量方面：1）中心质量得分是87.25分，目标是85分，全国排名14名，管控较好，请继续保持。2）进港超时库存率是6.06%，目标是5.2%，全国排名42名，管控较差，请继续努力。3）重复进线率是4.72%，目标是4%，全国排名57名，管控较差，请继续努力。4）始发破损率是十万分之30.24，目标是十万分之90，全国是十万分之83.26，全国4名，管控较好，请继续保持。东莞,目的 华北,破损量 51, 东莞,目的 佛山,破损量 45, 东莞,目的 揭阳,破损量 41, 5）倒包不彻底量是23票，请及时关注处理；3、成本方面：1）单票出港运能成本是0.331元，目标是0.368元，全国是0.395元,，全国排名11名，管控较好，请继续保持。东莞,目的中心 广西梧州市直营交换站,出港运能成本 2.43,出港量 1,单票成本 2.428, 东莞,目的中心 拉萨,出港运能成本 1740.75,出港量 1177,单票成本 1.479, 东莞,目的中心 内蒙古呼伦贝尔市交换站,出港运能成本 1000.23,出港量 684,单票成本 1.462, 2）单次操作成本是0.091元，目标是0.097元，全国是0.111元，全国排名17名，管控较好，请继续保持。4、效率方面：1）出港收件平均操作效率是1483件\/h，目标是1320件\/h，全国是1249件\/h，全国排名10名，管控较好，请继续保持。2）进港卸车平均操作效率是2084件\/h，目标是2000件\/h，全国是1440件\/h，全国排名4名，管控较好，请继续保持。3）进港拆包平均操作效率是2065票\/h，目标是2000票\/h，全国是1710票\/h，全国排名3名，管控较好，请继续保持。5、人员方面：1）小时工占比是0（≤5%），固定工出勤率是88.01%(≥86%)，人均效能是1542.57，目标是1420，评估是否需要招聘；2）人员流失率当月累计是15.01%，全国是6.6%，全国排名76名，管控较差，请继续努力。上个月是4.82%，环比趋势变差，需重点关注；中心操作储备,月初在职 12,月末在职 9,入职 0,离职 3,流失率 28.57, 中心操作,月初在职 0,月末在职 0,入职 0,离职 0,流失率 0, 国际仓操作区,月初在职 0,月末在职 0,入职 0,离职 0,流失率 0, 3）稳定性指数是47.5，全国是45.2，管控较好，请继续保持；中心操作储备,在职 9,平均工龄 10.4,稳定性指数 28.9, 6、安全与目视化：1）装车不规范：告警3次，请及时关注；2）月台下站人：告警7次，请及时关注；3）车等货监控：告警200次，请及时关注；4）拉包不码货告警0次，累计告警47次，请注意网点车位码货。	'
# text = '2024-07-30东莞中心综合诊断如下：1、综合评价：1）中心综合评价得分是88.17分，全国排名15名，管控较好，请继续保持。2、质量方面：1）中心质量得分是87.25分，目标是85分，全国排名14名，管控较好，请继续保持。2）进港超时库存率是6.06%，目标是5.2%，全国排名42名，管控较差，请继续努力。3）重复进线率是4.72%，目标是4%，全国排名57名，管控较差，请继续努力。4）始发破损率是十万分之30.24，目标是十万分之90，全国是十万分之83.26，全国4名，管控较好，请继续保持。'
# text = '周末我行走在湖边上看着湖面上航行的船只。'
# text = 'CCR读成 试试看, 贵阳-淮安,线路里程() 1776km。'
# text = "未来，圆通只人将做坚信之人，行分享之事，让圆通更智慧，让员工更幸福，让人生更精彩。"
text = '2024-08-01郑州中心进港诊断：1、超时库存：进港离场超时库存27243票，当前未解决11695票。379001,网点名称 河南省洛阳市,当日超时未解决票数 461, 371069,网点名称 河南省郑州市龙湖,当日超时未解决票数347， 371050,网点名称 河南省郑州市中牟县,当日超时未解决票数 308, 2、进港卸车：1）早班待卸车6辆，中班待卸车4辆；实时待卸车5辆，平均等待时长0.59h，等待时长超1小时1辆，待卸87419票，待卸7910件，待卸2846包。AQ951905006954,始发 -,等待时长 1.76, 2）进港卸车平均卸车效率1476.00。3）未来0-1小时到达车3辆，1231包，4463件单件，31061票。1-2小时2辆，916包，3442件单件，27307票。2小时以上49辆，26147包，102273件单件，1009847票。3、进港拆包：1）早班待拆包449个，中班待拆包177个，实时待拆包198个,平均滞留时长0.17h；滞留超2小时以上1包。NW21210156292,建包单位 河北省保定市高碑店市白沟镇,拆包单位 郑州转运中心,滞留时长(h) 3.37, 2）进港分拣操作票量804701，平均操作效率1757.00件\/h，最新操作效率1755.00件\/h。1-DWS-18,最新操作效率 837, 1-DWS-3,最新操作效率 1292, 1-DWS-15,最新操作效率 1304,3）进港小循环拥堵告警0次，累计告警1次，请注意控制小循环流量。4）进港未建包量712，当日累计未建包量8395，累计占比36.31%，请注意管控，严禁单件进入下包线。4、进港上车：1）进港错装55票，进港错装率0.31%。2）拉包不码货告警0次，累计告警4次，请注意网点车位码货。郑州,最新告警次数 1216,累计告警次数 38, 3）进港车等货累计告警92次。河南省三门峡市义马市,累计告警次数 28,最新告警次数 1, 河南省郑州市郑东新区职教园,累计告警次数 16,最新告警次数 1, 河南省郑州市金水东区,累计告警次数16,最新告警次数 1。'
# text = '语言，人们用来抒情达意；文字，人们用来记言记事。'
text = '根据研究对象的不同，环境物理学分为以下五个分支学科：──环境声学；──环境光学；──环境热学；──环境电磁学；──环境空气动力学。'

task_data['text'] = text


# asyncio.run(gen_tts(task_data))
# # asyncio.run(main(32))
run_gen_tts_in_process_pool(task_data, num_worker=1)

# if __name__ == '__main__':
#     text = sys.argv[1]
    
#     task_data['text'] = text
#     asyncio.run(gen_tts(task_data))
    
    
