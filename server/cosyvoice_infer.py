# -*- coding: utf-8 -*-
# @Author  : ZhangHang

import os
import resampy
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"
import torch
from hyperpyyaml import load_hyperpyyaml
import librosa
import torchaudio
import time
import random
import numpy as np
import wave
from tqdm import tqdm
import copy
import sys
from tts_front_cosyvoice import TTS_Front
from conf import COSYVOICE_ROOT_DIR, MATCHA_ROOT_DIR, COSYVOICE_MODEL_DIR, ErrorCode,use_deepgpu
from conf import SERVER_ROOT_DIR
from joblib import dump, load
from pickle import dumps, loads
import base64
from thop import profile
import time
import redis
import soundfile as sf
sys.path.append(COSYVOICE_ROOT_DIR)
sys.path.append(MATCHA_ROOT_DIR)
from datetime import datetime
from speaker_embedding import SpeakerEmbeddingRedis

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel

class CosyVoice:
    def __init__(self, model_dir, device,fp16,load_jit):
        self.model_dir = model_dir
        self.device = device
        self.model_name = "cosyvoice"
        with open(f"{model_dir}/cosyvoice.yaml", "r") as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            f"{model_dir}/campplus.onnx",
            f"{model_dir}/speech_tokenizer_v1.onnx",
            f"{model_dir}/spk2info.pt",
            False,
            configs["allowed_special"],
        )
        # self.frontend.spk2info['中文女']['embedding'] = torch.from_numpy( np.load('../resources/baoyi-tianmei.npy')[np.newaxis,:])
        self.tts_front = TTS_Front()

        self.model = CosyVoiceModel(configs["llm"], configs["flow"], configs["hift"], device,fp16)
        print(f'加载模型  fp16 = {fp16}  load_jit = {load_jit}')

        # 权重恢复
        self.model.load(f"{model_dir}/llm.pt", f"{model_dir}/flow.pt", f"{model_dir}/hift.pt")

        if load_jit:
            print(f'加载量化模型')
            self.model.load_jit('{}/llm.text_encoder.fp16.zip'.format(model_dir),
                                '{}/llm.llm.fp16.zip'.format(model_dir),
                                '{}/flow.encoder.fp32.zip'.format(model_dir))

        print(f"模型权重加载成功")
        self.spkEmbs = SpeakerEmbeddingRedis()
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id):
        tts_speeches = []
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_sft(i, spk_id)
            model_output = self.model.inference(**model_input)
            tts_speeches.append(model_output["tts_speech"])
        return {"tts_speech": torch.concat(tts_speeches, dim=1)}

    def inference_prompt_text_audio(self, prompt_text, prompt_speech):
        prompt_speech_16k = cosyvoice.load_wav(prompt_speech)
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        speaker_info = self.frontend.frontend_zero_shot_speaker(
            prompt_text, prompt_speech_16k
        )
        return speaker_info

    def inference_zero_shot(self, tts_text, speaker_info):
        tts_texts = self.tts_front.process(tts_text, split=True)
        tts_text_token, tts_text_token_len = self.frontend.frontend_tts_text(tts_text)

        model_input = copy.deepcopy(speaker_info)
        model_input["text"] = tts_text_token
        model_input["text_len"] = tts_text_token_len
        tts_speech = self.model.inference(**model_input)

        return tts_speech

    def generate_spk_emb(self, prompt_text, prompt_wav):
        top_db = 60
        hop_length = 220
        win_length = 440
        target_sr = 16000

        prompt_speech, orig_freq = librosa.load(prompt_wav, sr=None)
        # 移除音频前后静音段
        speech, _ = librosa.effects.trim(
            prompt_speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
        )
        # 音频音量超过阈值，则统一缩放
        max_val = 0.8
        if np.max(np.abs(speech)) > max_val:
            speech = speech / np.max(np.abs(speech)) * max_val
        # 音频末尾补0
        # speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
        speech = np.concatenate([speech, np.zeros(int(target_sr * 0.2))])

        prompt_speech_16 = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=target_sr)(torch.from_numpy(prompt_speech)).numpy()
        prompt_speech_22 = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=22050)(torch.from_numpy(prompt_speech)).numpy()

        prompt_speech_16 = torch.from_numpy( prompt_speech_16[np.newaxis, :])
        prompt_speech_22 = torch.from_numpy( prompt_speech_22[np.newaxis, :])

        # 提取音色、参考文本特征
        speaker_info = self.frontend.frontend_zero_shot_speaker_v2(prompt_text, prompt_speech_16,prompt_speech_22)

        cpu_device = torch.device("cpu")
        for key in speaker_info.keys():
            speaker_info[key] = speaker_info[key].to(cpu_device)

        spk_info = {}
        spk_info["speech"] = speech
        spk_info["norm_text"] = prompt_text
        spk_info[self.model_name] = speaker_info
        return spk_info

    def clone_infer(self, tts_text, spk_data):
        speaker_info = spk_data[self.model_name]
        tts_text_token, tts_text_token_len = self.frontend.frontend_tts_text(tts_text)

        model_input = {}
        for key in speaker_info.keys():
            model_input[key] = speaker_info[key].to(self.device)

        model_input["text"] = tts_text_token
        model_input["text_len"] = tts_text_token_len
        set_all_random_seed(0)

        # 计算FLOPS
        # flops, params = profile(model, inputs=(input,), verbose=False)
        # # 输出结果
        # print(f"Model FLOPS: {flops / (1000 ** 3):.2f} GFLOPS")

        tts_speech = self.model.inference(**model_input)

     
        audio, audio_sample_rate, datas = self.post_audio(tts_speech,tts_text)
        return  audio, audio_sample_rate, datas

    def inference_instruct(self, tts_text, spk_id, instruct_text='You are a female speaker with a slightly faster speaking pace, normal tone, and stable emotions.'):
        model_input = self.frontend.frontend_instruct(tts_text, spk_id, instruct_text)
        tts_speech = self.model.inference(**model_input)
        audio, audio_sample_rate, datas = self.post_audio(tts_speech,tts_text)
        return  audio, audio_sample_rate, datas

    def post_audio(self,tts_speech,tts_text):
        audio = tts_speech.numpy().flatten()
        audio_sample_rate = 22050

        # 增加静音隔断
        zero_wav = np.zeros(int(audio_sample_rate * 0.2), dtype=np.float32)
        final_audio = np.concatenate([audio, zero_wav], 0)

        # 字幕
        dur = len(audio) / audio_sample_rate
        num_wd = len(tts_text)
        step = dur / num_wd
        # datas = []
        # for i, wd in enumerate(tts_text):
        #     datas.append([wd, i * step, (i + 1) * step])
        datas = [[wd, i * step, (i + 1) * step] for i, wd in enumerate(tts_text)]
        return audio, audio_sample_rate, datas
def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_wav(audio_data):
    # 参数设置
    channels = 1  # 单声道
    sample_width = 2  # 量化位数，这里使用16位

    # 将 float32 数据转换为 int16
    # audio_data_int16 = (audio_data * (2 ** 15 - 1)).astype(np.int16)

    # 创建 WAV 文件
    with wave.open("../resources/out_clone.wav", "wb") as wf:
        # 设置 WAV 文件参数
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(22050)
        # 写入音频数据
        wf.writeframes(audio_data.tobytes())


def get_gpu_info(gpu_index):
    handle = nvmlDeviceGetHandleByIndex(gpu_index)

    # 获取GPU名称
    name = nvmlDeviceGetName(handle)
    log_str = f"GPU {gpu_index}: {name}\n"

    # 获取GPU温度
    temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    log_str += f"  Temperature: {temperature} °C\n"

    # 获取GPU功耗
    power_draw = nvmlDeviceGetPowerUsage(handle)
    power_draw_watts = power_draw / 1000  # 转换为瓦特
    log_str += f"  Power Draw: {power_draw_watts:.2f} W\n"

    # 获取GPU利用率
    utilization = nvmlDeviceGetUtilizationRates(handle)
    log_str += f"  Utilization: {utilization.gpu}%\n"

    # 获取GPU显存信息
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    log_str += f"  Total Memory: {mem_info.total / 1024 ** 2:.2f} MB\n"
    log_str += f"  Used Memory: {mem_info.used / 1024 ** 2:.2f} MB\n"
    log_str += f"  Free Memory: {mem_info.free / 1024 ** 2:.2f} MB\n"
    return log_str

def write_gpu_info():
    with open("../resources/gpu.txt", "w") as f:
        for index in range(100000000):
            start_time = time.time()
            texts = tts_front.process(tts_text, split=True)
            audios = []
            print(f"input_text = {tts_text}")
            for text in texts:
                print(f'cur_text = {text}')
                audio, fs, stamp = cosyvoice.clone_infer(
                    tts_text=text, spk_data=spk_data
                )
                audios += list(audio)

            audio_time = len(audios) / 22050
            end_time = time.time()
            rtf_value = (end_time - start_time) / audio_time
            # print(f"audio_time = {audio_time}, rtf_value = {rtf_value}")

            now = datetime.now()
            formatted_time_1 = now.strftime("%Y%m%d%H%M%S")
            gpu_info_str = get_gpu_info(gpu_index)
            # print(f'gpu_info  =  {gpu_info_str}')

            log_str = f"current_time = {formatted_time_1}     index = {index+1}   \n"
            log_str += f"gpu_info：\n{gpu_info_str}"
            log_str += f"audio_time = {audio_time}, rtf_value = {rtf_value}\n"
            log_str += f"cut_text = {texts}\n\n"
            print(log_str)
            f.write(log_str)
    nvmlShutdown()

def save_spk_info(prompt_wav,prompt_text,spk_id):
    # rds = redis.Redis(host="127.0.0.1", port=6379, db=0)
    spkEmbs = SpeakerEmbeddingRedis()

    spk_data = cosyvoice.generate_spk_emb(
        prompt_text=prompt_text, prompt_wav=prompt_wav
    )
    # serialized_data = dumps(spk_data)
    # encoded_data = base64.b64encode(serialized_data).decode('utf-8')
    # rds.set(spk_id, encoded_data)
    spkEmbs.save_spk_data(spk_id, spk_data)
    return spk_data

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cosyvoice = CosyVoice(COSYVOICE_MODEL_DIR, device=device,fp16=True,load_jit=True)
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # serialized_data = rds.get(spk_id)
    # decoded_data = base64.b64decode(serialized_data)
    # spk_data = loads(decoded_data)

    # prompt_wav = f"{root_path}/resources/baoyi.wav"
    # prompt_text = "你们不仅是圆通国际业务的拓荒者,更要做圆通国际业务的奠基者.在3月15日圆通国际精彩计划见面会上,面对首期入选人员,这既是董事长喻渭蛟的真挚祝福,更是他对圆通走向国际的殷切期望."
    # spk_id = "cosyvoice-baoyi"
    # spk_data = save_spk_info(prompt_wav,prompt_text,spk_id)

    # prompt_wav = f"{root_path}/resources/weiwei.wav"
    # prompt_text = "二零二四年八月三十日滇西中心质量分析如下:一,中心质量:一中心质量得分是九十一点三八分."
    # spk_id = "cosyvoice-weiwei"
    # spk_data = save_spk_info(prompt_wav,prompt_text,spk_id)

    # prompt_wav = f"{root_path}/resources/weiwei_tianmei.wav"
    # prompt_text = "二零二四年八月三十日滇西中心质量分析如下:一,中心质量:一中心质量得分是九十一点三八分、"
    # spk_id = "cosyvoice-weiwei-tianmei"
    # spk_data = save_spk_info(prompt_wav,prompt_text,spk_id)

    # prompt_wav = f"{root_path}/resources/luoying.wav"
    # prompt_text = "二零二四年八月三十日滇西中心质量分析如下:一,中心质量:一中心质量得分是九十一点三八分."
    # spk_id = "cosyvoice-luoying"
    # spk_data = save_spk_info(prompt_wav,prompt_text,spk_id)
 
    # prompt_wav = f"{root_path}/resources/ranran.wav"
    # prompt_text = "重复进线率减少超百分之五十,服务质量稳步提升.圆通速递定位于互联网信息技术的快递平台."
    # spk_id = "cosyvoice-ranran"
    # spk_data = save_spk_info(prompt_wav,prompt_text,spk_id)


    # prompt_wav = f"{root_path}/resources/cancan.wav"
    # prompt_text = "二零二四年九月六日,上海网点重复进线率减少超百分之五十,服务质量稳步提升.圆通速递定位于互联网信息技术的快递平台."
    # spk_id = "cosyvoice-cancan"
    # spk_data = save_spk_info(prompt_wav,prompt_text,spk_id)
    
    # prompt_wav = f"{root_path}/resources/xiaoyuan.wav"
    # prompt_text = "二零二四年九月六日,上海网点重复进线率减少超百分之五十,服务质量稳步提升.圆通速递定位于互联网信息技术的快递平台."
    # spk_id = "cosyvoice-xiaoyuan"
    # spk_data = save_spk_info(prompt_wav,prompt_text,spk_id)

    prompt_wav = f"{root_path}/resources/weiwei_tianmei.wav"
    prompt_text = "二零二四年八月三十日滇西中心质量分析如下:一,中心质量:一中心质量得分是九十一点三八分、"
    spk_id = "instruct"
    spk_data = save_spk_info(prompt_wav,prompt_text,spk_id)
    
    print('音色载入完成')
    # 目标文本合成音频
    tts_text = "未来，圆通只人将做坚信之人，行分享之事，让圆通更智慧，让员工更幸福，让人生更精彩。"
    tts_front = TTS_Front()
    texts = tts_front.process(tts_text, split=True)


    audios = []
    print(f"input_text = {tts_text}")
    for text in texts:
        print(f'cur_text = {text}')
        # audio, fs, stamp = cosyvoice.clone_infer(tts_text=text, spk_data=spk_data)
        audio, fs, stamp = cosyvoice.inference_instruct(text, spk_id='中文女')
        audios += list(audio)

    audios = np.array(audios)
    sf.write(f'../resources/out_clone.wav', audios, 22050, subtype='PCM_16')
