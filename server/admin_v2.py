import os
os.environ['version'] = 'v2'


import gradio as gr
import requests
import json
import websockets
import librosa
import torch
import sys
import numpy as np
import copy
import traceback
import ffmpeg
import logging
import soundfile as sf

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s-%(name)s-%(filename)s-%(funcName)s-%(lineno)d-%(levelname)s-%(process)d] %(message)s')
logger = logging.getLogger(__name__)

global g_audio, g_fs
g_audio = None
g_fs = None

random_seed = 12354
np.random.seed(random_seed)  # 设置 NumPy 的随机种子为 42
torch.manual_seed(random_seed)  # 为 CPU 设置随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)  # 为所有 GPU 设置随机种子
    
rm_path = '/home/zhangjiayuan/program/audio/tts/GPT-SoVITS'
rm_paths = []
ori = copy.deepcopy(sys.path)
for pth in ori:
    if rm_path in pth:
        print(f'rm sys path: {pth}')
        sys.path.remove(pth)


from conf import GPT_SOVITS_MODEL_DIR, SERVER_ROOT_DIR
sys.path.append(SERVER_ROOT_DIR)
from gpt_sovits_model_v2 import  zoro_gpt_sovits  
from speaker_embedding import SpeakerEmbeddings

from tts_front import TTS_Front

from audio_post_process import audio_post_process

tts_front = TTS_Front()

# tts_front.logger.propagate = True 

device=torch.device('cuda:2')
tts_model = zoro_gpt_sovits(device=device)
# spk_embs = SpeakerEmbeddings()
spk_embs = tts_model.tts_spk_embs
# spk_embs.load_spk_embs()

gpt_models, sovits_models = tts_model.get_models()
gpt_model_names = sorted(gpt_models.keys())
sovits_model_names = sorted(sovits_models.keys())

global default_spk_names
default_spk_names = list(set(spk_embs.get_spk_names()))

print(f'adminv2---')
print(sys.path)

def do_audio_post_process(post_fs, volume_ratio, speed_ratio, pitch_ratio):
    global g_audio, g_fs
    logger.info(f'ori audio shape: {g_audio.shape}, dtype {g_audio.dtype}, fs {g_fs}, post_fs {post_fs}')
    post_audio = audio_post_process(g_audio, g_fs, int(post_fs), float(volume_ratio), float(speed_ratio), float(pitch_ratio))
    yield int(post_fs), post_audio

def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def flush_spk_names():
    global default_spk_names
    default_spk_names = spk_embs.get_spk_names()
    return {"choices": default_spk_names, "__type__": "update"}

def update_model_list():
    gpt_models, sovits_models = tts_model.get_models()
    gpt_model_names = sorted(gpt_models.keys())
    sovits_model_names = sorted(sovits_models.keys())
    return {"choices": gpt_model_names, "__type__": "update"}, {"choices": sovits_model_names, "__type__": "update"}

def change_sovits_weights(sovits_name):
    tts_model.load_sovits_weights(sovits_models[sovits_name])
    
def change_gpt_weights(gpt_name):
    tts_model.load_gpt_weights(gpt_models[gpt_name])

def do_tts_front(input_tts_text):
    input_tts_text_split = tts_front.process(input_tts_text)
    res = "\n".join(input_tts_text_split)
    print(f'----tts_front---')
    print(res)
    print(f'-----tts_front end----')
    return res

def synthesize_default_voice(spk_name, input_tts_text):
    ref_spk_data = spk_embs.get_spk_emb(spk_name)
    
    input_tts_text_split = tts_front.process(input_tts_text)
    final_audio = []
    for text in input_tts_text_split:
        logger.info(f'---infer seg text: {text}')
        audio, fs, stamp = tts_model.infer(text, 'zh', ref_spk_data)
        final_audio.append(audio)
    final_audio = np.concatenate(final_audio)
    if final_audio.dtype != np.int16:
        final_audio = (final_audio*32768).astype(np.int16)
    sf.write('out_wav_admin_v2.wav', final_audio, fs, subtype='PCM_16')
    global g_audio, g_fs
    g_audio = final_audio
    g_fs = fs
    yield fs, final_audio

def synthesize_custom_voice(ref_spk_name, ref_wav_file, ref_text, top_k, top_p, temperature, input_tts_text):
    global default_spk_names
    if ref_spk_name in default_spk_names:
        raise OSError("音色名称已存在，重新输入音色名称")

    print(f'clone ref_text {ref_text} tts_text {input_tts_text}')
    
    audio_spec = load_audio(ref_wav_file, int(32000))
    wav16k, sr = librosa.load(ref_wav_file, sr=16000)

    # if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
    #     raise OSError("参考音频在3~10秒范围外，请更换！")
    
    ref_spk_data = tts_model.generate_spk_emb_admin(audio_spec, wav16k, fs=32000, text=ref_text, language="zh")
    # input_tts_text_split = tts_front.process(input_tts_text)
    input_tts_text_split = input_tts_text.split('\n')
    final_audio = []
    for text in input_tts_text_split:
        logger.info(f'---infer seg text: {text}')
        audio, fs, stamp = tts_model.infer(text, 'zh', ref_spk_data, top_k=top_k, top_p=top_p, temperature=temperature)
        final_audio.append(audio)
    final_audio = np.concatenate(final_audio)
    if final_audio.dtype != np.int16:
        final_audio = (final_audio*32768).astype(np.int16)
    sf.write('out_wav_admin_v2.wav', final_audio, fs, subtype='PCM_16')
    global g_audio, g_fs
    g_audio = final_audio
    g_fs = fs
    yield fs, final_audio
    

def save_speaker_embs(ref_spk_name, ref_wav_file, ref_text):
    global default_spk_names
    if ref_spk_name in default_spk_names:
        raise OSError("音色名称已存在，重新输入音色名称")
    
    wav16k, sr = librosa.load(ref_wav_file, sr=16000)
    wav32k, sr2 = librosa.load(ref_wav_file, sr=32000)
    # if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
    #     raise OSError("参考音频在3~10秒范围外，请更换！")
    
    # ref_spk_data = tts_model.generate_spk_emb(wav16k, text=ref_text, language="zh")
    ref_spk_data = tts_model.generate_spk_emb_admin(wav32k, wav16k, fs=sr2, text=ref_text, language="zh")

    spk_embs.save_spk_emb(ref_spk_name, ref_spk_data)
    default_spk_names = spk_embs.get_spk_names()
    

with gr.Blocks(title="GPT-SoVITS-V2 WebUI") as app:

    with gr.Tabs() as spk_tabs:
        with gr.TabItem('语音合成-v2'):
            with gr.Row():
                default_spk_name = default_spk_names[0] if len(default_spk_names) > 0 else ""
                spk_dropdown = gr.Dropdown(label="音色列表", choices=default_spk_names, value=default_spk_name, interactive=True)
                flush_spk_button = gr.Button("刷新音色列表", variant="primary")
                flush_spk_button.click(fn=flush_spk_names, inputs=[], outputs=[spk_dropdown])
            with gr.Row():
                input_tts_text = gr.Textbox(label="输入合成文本", value="", interactive=True)
                inference_button = gr.Button("合成语音", variant="primary")
            with gr.Row():
                output = gr.Audio(label="输出的语音")
            inference_button.click(fn=synthesize_default_voice, inputs=[spk_dropdown, input_tts_text], outputs=[output])
            
        with gr.TabItem('语音克隆-v2'):
            gr.Markdown(value="模型切换")
            with gr.Row():
                GPT_dropdown = gr.Dropdown(label="GPT模型列表", choices=gpt_model_names, value=gpt_model_names[0], interactive=True)
                SoVITS_dropdown = gr.Dropdown(label="SoVITS模型列表", choices=sovits_model_names, value=sovits_model_names[0], interactive=True)
                refresh_button = gr.Button("刷新模型路径", variant="primary")
                refresh_button.click(fn=update_model_list, inputs=[], outputs=[GPT_dropdown, SoVITS_dropdown])
                SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown], [])
                GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])
            with gr.Row():
                ref_spk_name = gr.Textbox(label="音色名称", value="", interactive=True)
            with gr.Row():
                top_k = gr.Slider(minimum=1,maximum=100,step=1,label="top_k",value=5,interactive=True)
                top_p = gr.Slider(minimum=0,maximum=1,step=0.05,label="top_p",value=1,interactive=True)
                temperature = gr.Slider(minimum=0,maximum=1,step=0.05,label="temperature",value=1,interactive=True) 
            with gr.Row():
                ref_wav_file = gr.Audio(label="请上传3~10秒内参考音频，超过会报错！", type="filepath", interactive=True)
                ref_text = gr.Textbox(label="参考音频文本内容", value="", interactive=True)
            with gr.Row():
                save_spk_button = gr.Button("保存音色", variant="primary")
                save_spk_button.click(fn=save_speaker_embs, inputs=[ref_spk_name, ref_wav_file, ref_text], outputs=[])
            with gr.Row():
                input_tts_text = gr.Textbox(label="输入合成文本", value="", interactive=True)
                tts_front_button = gr.Button("正则切分", variant="primary")
                output_tn_text = gr.Textbox(label="正则输出文本", value="", interactive=True)
                tts_front_button.click(fn=do_tts_front, inputs=[input_tts_text], outputs=[output_tn_text])
                
            with gr.Row():
                clone_button = gr.Button("合成语音", variant="primary")
            with gr.Row():
                output = gr.Audio(label="输出的语音")
            clone_button.click(fn=synthesize_custom_voice, 
                                    inputs=[ref_spk_name, ref_wav_file, ref_text, top_k, top_p, temperature, output_tn_text], 
                                    outputs=[output])
    
    gr.Markdown(value="音频后处理")
    with gr.Row():
        post_fs = gr.Textbox(label="输出采样率", value="", interactive=True)
        volume_ratio = gr.Slider(minimum=0,maximum=2,step=0.05,label="音量",value=1,interactive=True) 
        speed_ratio = gr.Slider(minimum=0,maximum=2,step=0.05,label="语速",value=1,interactive=True)
        pitch_ratio = gr.Slider(minimum=0,maximum=2,step=0.05,label="音高",value=1,interactive=True)
    with gr.Row():
        audio_post_button = gr.Button("后处理", variant="primary")
    with gr.Row():
        audio_post_output = gr.Audio(label="输出的语音")
    audio_post_button.click(fn=do_audio_post_process, inputs=[post_fs, volume_ratio, speed_ratio, pitch_ratio], outputs=[audio_post_output])






if __name__ == '__main__':
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        server_port=7101,
        quiet=True,
    )