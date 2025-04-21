import hashlib
import logging
import os
import re
import tempfile
import time

import httpx

from conf import SERVER_ROOT_DIR

# from fastapi import FastAPI, UploadFile, File

log_root_dir = f'{SERVER_ROOT_DIR}/logs'
md5_hash = hashlib.md5()

os.makedirs(log_root_dir, exist_ok=True)

formatter = logging.Formatter('[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s')
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s')



def init_file_logger(filename, level = logging.DEBUG, propagate=True):

    logger = logging.getLogger(filename)
    logger.setLevel(level)  # 设置日志级别为DEBUG

    # 创建一个文件处理器，用于写入日志文件
    file_handler = logging.FileHandler(f'{log_root_dir}/{filename}.log')  # 日志文件路径
    file_handler.setLevel(level)  # 设置文件处理器的日志级别
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到logger中
    logger.addHandler(file_handler)
    logger.propagate = propagate 
    
    return logger



def read_scp(filename):
    utts = []
    wav_scp = {}
    
    with open(filename, 'r') as f:
        for line in f.readlines():
            buff = line.strip().split()
        
            if len(buff) != 2:
                logging.warning(f'{filename} num col={len(buff)} not equal expect-2 line: [{buff}]')
                exit(0)
            utt = buff[0]
            wav = buff[1]
            if utt not in wav_scp.keys():
                utts.append(utt)
                wav_scp[utt] = wav
            else:
                logging.warning(f'{filename} utt {utt} has been exist!')
    return utts, wav_scp


def count_num_wd(text):
    no_punct = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    
    pattern = r'[\u4e00-\u9fff]+|(?i)\b[a-zA-Z]+\b'
    # pattern = r'[\u4e00-\u9fff]+|\b[a-zA-Z]+\b'
    matches = re.findall(pattern, no_punct)
    
    count = 0    
    for match in matches:
        if re.match(r'[\u4e00-\u9fff]+', match):
            count += len(match)
        elif re.match(r'\b[a-zA-Z]+\b', match):
            count += 1

    return count



def speech_rtf_decorator(speech_idx, fs=16000):
    def decorator(func):
        def wrapper(*args, **kwargs):
            speech = args[speech_idx]
            audio_duration = len(speech) / fs
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            processing_time = end_time - start_time
            real_time_factor = processing_time / audio_duration
            logging.info(f"{func.__name__} process {audio_duration:.2f}s speech  use {processing_time:.2f} , RTF: {real_time_factor:.2f}/s")
            return result
        return wrapper
    return decorator


def text_rtf_decorator(text_idx):
    def decorator(func):
        def wrapper(*args, **kwargs):
            text = args[text_idx]
            num_wd = count_num_wd(text)
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            processing_time = end_time - start_time
            real_time_factor = num_wd / processing_time 
            logging.info(f"{func.__name__} process {num_wd} words  use {processing_time:.2f} , RTF: {real_time_factor:.2f}/s")
            return result
        return wrapper
    return decorator


def use_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        processing_time = end_time - start_time
        logging.info(f"{func.__name__} use {processing_time:.2f}s")
        return result
    return wrapper



def download_file(url, target_file="", md5=''):
    try:
        # 发起 GET 请求
        response = httpx.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # 保存文件
        if target_file=="":
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response.content)
                target_file = temp_file.name
        else:
            with open(target_file, "wb") as f:
                f.write(response.content)
        
        
        if md5 != '':
            md5_hash.update(response.content)
            md5_val = md5_hash.hexdigest()
            if md5_val != md5:
                logging.warning(f"download file {target_file} from {url} error, md5 {md5_val} not expect {md5}")
                return ""

        print(f"File downloaded successfully to {target_file}")
    except httpx.HTTPError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
    return target_file