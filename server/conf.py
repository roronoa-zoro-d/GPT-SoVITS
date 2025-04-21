import sys
import os

cur_file_path = os.path.abspath(__file__)
cur_file_dir = os.path.dirname(cur_file_path)
print(f'cur_file_path:{cur_file_path}')
print(f'cur_file_dir:{cur_file_dir}, {os.path.dirname(cur_file_dir)}')

TTS_ROOT_DIR = os.path.dirname(cur_file_dir)
use_deepgpu = False
use_instruct = False

GPT_ROOT_DIR = f'{TTS_ROOT_DIR}/GPT-SoVITS/GPT_SoVITS'
GPT_V2_ROOT_DIR = f'{TTS_ROOT_DIR}/GPT-SoVITS-v2-240807/GPT_SoVITS'
SERVER_ROOT_DIR = f'{TTS_ROOT_DIR}/server'
WETEXT_ROOT_DIR = f'{TTS_ROOT_DIR}/WeTextProcessing'

if use_deepgpu:
    COSYVOICE_ROOT_DIR = f'{TTS_ROOT_DIR}/CosyVoiceAli'
    print(f'使用deepgpu加速')
else:
    COSYVOICE_ROOT_DIR = f'{TTS_ROOT_DIR}/CosyVoice/'
    print(f'不使用deepgpu加速')
MATCHA_ROOT_DIR = f'{TTS_ROOT_DIR}/CosyVoice/third_party/Matcha-TTS/'
COSYVOICE_ROOT_DIR_V1 = f'{TTS_ROOT_DIR}/cosyvoice_v1'

# model_dir = '/root/workspace/models/'
model_dir = '/data/nas/rq/tts/cosyvoice_model'
GPT_SOVITS_MODEL_DIR = model_dir

# COSYVOICE_MODEL_DIR = f'{model_dir}/CosyVoice-300M-Instruct'
COSYVOICE_MODEL_DIR = f'{model_dir}/CosyVoice-300M'
# COSYVOICE_MODEL_DIR = f'{model_dir}/CosyVoiceModel_v1'


# 定义错误码
class ErrorCode:
    REDIS_SPKID = 10000  # redis 里面不存在spk_id对应的数据，或者数据为
    NO_KEY_IN_TASK = 10001  # task 中缺少 key 值
    OTHER_ERROR = 11000  # 其他错误，


class Punctuation(object):
    punc_map = {
        '，': ',',
        '：': ':',
        '；': ';',
        '？': '?',
        '!': '!',
        '＜': '<',
        '℃': '°C',
        '……': ',',
        '（': '(',
        '）': ')',
        '—': '-',
        '=': '=,'
    }
    # 按序号切分(语义层面切分)
    order_number_prefix = "[：。， ]"
    order_number_suffix = "[、）]"

    # 按标点切分, 
    split_punc = [
        "。！？",
        "；，=,;"
    ]
    # 特殊字符 只为统计
    special_punc = set("＞（-≥、*：≤/%‱＜）。，.")

    # (km) 括号里面包含这些字符，不读，则删除
    bracket_punc = ['km', 'h', '%', '‱']

    # 后处理模块删除的符号
    post_remove_punc = ['"', '(', ')', '《', '》', '/']


serial_number_map = {
    '0': '零',
    '1': '幺',
    '2': '二',
    '3': '三',
    '4': '四',
    '5': '五',
    '6': '六',
    '7': '七',
    '8': '八',
    '9': '九'
}
