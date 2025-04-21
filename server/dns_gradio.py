import sys
import os
import logging

import gradio as gr
import librosa
import soundfile as sf
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

import math
import numpy as np

sys.path.append('/home/zhangjiayuan/program/audio/DNS/dns/dtln_pytorch')
from realtime_onnx import load_model, run_dtln

def all_file_exist(files):
    for file in files:
        if os.path.exists(file) is False:
            return False
    return True

class DNS_Dtln(object):
    def __init__(self, model_dir=None, exp=14, model_idx=9):
        exp_dir = '/home/zhangjiayuan/program/audio/DNS/dns/dtln_pytorch/experiment/'
        if model_dir is None:
            self.model_dir = '/home/zhangjiayuan/program/audio/DNS/dns/dtln_pytorch/pretrained/'
            self.model1_path = f'{self.model_dir}/model_p1.onnx'
            self.model2_path = f'{self.model_dir}/model_p2.onnx'
        else:
            self.model_dir = f'{exp_dir}/exp{exp}/onnx_model/'
            self.model1_path = f'{self.model_dir}/model{model_idx}.pt_p1.onnx'
            self.model2_path = f'{self.model_dir}/model{model_idx}.pt_p2.onnx'
        
        if all_file_exist([self.model1_path, self.model2_path]) is False:
            logger.error(f'dtln model not exist: for {self.model1_path}')
            return None
        
        self.model = load_model(self.model1_path, self.model2_path)
        

    def __call__(self, speech):
        dns_speech = run_dtln(self.model, speech)
        return dns_speech
    

def dns_speech(exp_idx, model_idx, speech_path):
    print(f'dtln model exp-{exp_idx} inx-{model_idx}')
    dns = DNS_Dtln(exp=exp_idx, model_idx=model_idx)
    wav16k, sr = librosa.load(speech_path, sr=16000)
    dns_speech = dns(wav16k)
    return sr, dns_speech

with gr.Blocks(title="DTLN") as app:
    with gr.Row():
        exp_idx = gr.Textbox(label="模型exp索引", value="14", interactive=True)
        model_idx = gr.Textbox(label="模型索引", value="9", interactive=True)
        input_audio = gr.Audio(label="Input Audio", type="filepath", interactive=True)
        dns = gr.Button("降噪", variant="primary")
    with gr.Row():
        output = gr.Audio(label="降噪后输出的语音")
    with gr.Row():
        dns.click(fn=dns_speech, inputs=[exp_idx, model_idx, input_audio], outputs=[output])
        



if __name__ == '__main__':
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        server_port=7102,
        quiet=True,
    )