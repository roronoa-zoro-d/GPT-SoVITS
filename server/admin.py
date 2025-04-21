import gradio as gr
import requests
import json
import websockets
import librosa
import torch
import sys
import numpy as np
import copy

rm_path = '/home/zhangjiayuan/program/audio/tts/GPT-SoVITS'
rm_paths = []
ori = copy.deepcopy(sys.path)
for pth in ori:
    if rm_path in pth:
        print(f'rm sys path: {pth}')
        sys.path.remove(pth)


from conf import GPT_SOVITS_MODEL_DIR, SERVER_ROOT_DIR
sys.path.append(SERVER_ROOT_DIR)
from gpt_sovits_model import  zoro_gpt_sovits  
from speaker_embedding import SpeakerEmbeddings

from tts_front import TTS_Front

tts_front = TTS_Front()

device=torch.device('cuda:2')
tts_model = zoro_gpt_sovits(device=device)
spk_embs = SpeakerEmbeddings()
# spk_embs.load_spk_embs()

gpt_models, sovits_models = tts_model.get_models()
gpt_model_names = sorted(gpt_models.keys())
sovits_model_names = sorted(sovits_models.keys())

global default_spk_names
default_spk_names = spk_embs.get_spk_names()

print(f'admin---')
print(sys.path)

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


def synthesize_default_voice(spk_name, input_tts_text):
    ref_spk_data = spk_embs.get_spk_emb(spk_name)
    audio, fs, stamp = tts_model.infer(input_tts_text, "zh", ref_spk_data)
    yield fs, audio

def synthesize_custom_voice(ref_spk_name, ref_wav_file, ref_text, top_k, top_p, temperature, input_tts_text):
    global default_spk_names
    if ref_spk_name in default_spk_names:
        raise OSError("音色名称已存在，重新输入音色名称")

    print(f'clone ref_text {ref_text} tts_text {input_tts_text}')

    wav16k, sr = librosa.load(ref_wav_file, sr=16000)
    # if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
    #     raise OSError("参考音频在3~10秒范围外，请更换！")
    
    ref_spk_data = tts_model.generate_spk_emb(wav16k, text=ref_text, language="zh")
    input_tts_text_split = tts_front.process(input_tts_text)
    final_audio = []
    for text in input_tts_text_split:
        audio, fs, stamp = tts_model.infer(text, 'zh', ref_spk_data)
        final_audio.append(audio)
    final_audio = np.concatenate(final_audio)
    yield fs, final_audio
    

def save_speaker_embs(ref_spk_name, ref_wav_file, ref_text):
    global default_spk_names
    if ref_spk_name in default_spk_names:
        raise OSError("音色名称已存在，重新输入音色名称")
    
    wav16k, sr = librosa.load(ref_wav_file, sr=16000)
    # if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
    #     raise OSError("参考音频在3~10秒范围外，请更换！")
    
    ref_spk_data = tts_model.generate_spk_emb(wav16k, text=ref_text, language="zh")

    spk_embs.save_spk_emb(ref_spk_name, ref_spk_data)
    default_spk_names = spk_embs.get_spk_names()
    

with gr.Blocks(title="GPT-SoVITS WebUI") as app:

    with gr.Tabs() as spk_tabs:
        with gr.TabItem('语音合成'):
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
            
        with gr.TabItem('语音克隆'):
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
            with gr.Row():
                clone_button = gr.Button("合成语音", variant="primary")
            with gr.Row():
                output = gr.Audio(label="输出的语音")
            clone_button.click(fn=synthesize_custom_voice, 
                                    inputs=[ref_spk_name, ref_wav_file, ref_text, top_k, top_p, temperature, input_tts_text], 
                                    outputs=[output])






if __name__ == '__main__':
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        server_port=7100,
        quiet=True,
    )