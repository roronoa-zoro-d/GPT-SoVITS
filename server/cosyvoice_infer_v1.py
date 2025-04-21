# -*- coding: utf-8 -*-
# @Author  : ZhangHang

import copy
import os
import random
import sys
import time

import inflect
import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
import whisper.tokenizer
from omegaconf import DictConfig
from tqdm import tqdm

from conf import COSYVOICE_MODEL_DIR, TTS_ROOT_DIR,COSYVOICE_ROOT_DIR_V1
sys.path.append(TTS_ROOT_DIR)
sys.path.append(COSYVOICE_ROOT_DIR_V1)

from tts_front_cosyvoice import TTS_Front

from cosyvoice_v1.cosyvoice_model import TransformerEncoder, LegacyLinearNoSubsampling, EspnetRelPositionalEncoding, \
    RelPositionMultiHeadedAttention, LinearNoSubsampling, ConformerEncoder, ConditionalDecoder, MaskedDiffWithXvec, \
    InterpolateRegulator, ConditionalCFM, HiFTGenerator, ConvRNNF0Predictor
from cosyvoice_v1.llm import TransformerLM
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}
class CosyVoiceInfer:
    def __init__(self, model_dir, fp16=False, use_jit=False, device='cpu'):
        self.model_dir = model_dir
        # spk2info_file = '/data/nas/rq/tts/cosyvoice_model/llm_data_v2/spk2embedding.pt'
        spk2info_file = f'{model_dir}/spk2info.pt'

        self.frontend = CosyVoiceFrontEnd(f'{model_dir}/campplus.onnx', f'{model_dir}/speech_tokenizer_v1.onnx',
                                          spk2info_file)
        self.model = CosyVoiceModel(model_dir=model_dir, fp16=fp16, use_jit=use_jit, use_onnx=False, device=device)
        self.model.load(f'{model_dir}/llm.pt', f'{model_dir}/flow.pt', f'{model_dir}/hift.pt')

    def inference_spk(self, prompt_text, prompt_speech_16k):  # 提取prompt音色信息
        self.model_input_base = self.frontend.clone_spk(prompt_text, prompt_speech_16k)

    def inference_sft(self, tts_text, spk_id):  # 官方音色文本合成
        model_input = self.frontend.frontend_sft(tts_text, spk_id)
        tts_speech = self.model.inference(**model_input)["tts_speech"]
        return self.post_audio(tts_speech, tts_text)

    def inference_prompt_text_audio(self, tts_text):  # zero-shot合成
        set_all_random_seed(0)
        model_input = self.frontend.clone_text(tts_text, self.model_input_base)
        tts_speech = self.model.inference(**model_input)["tts_speech"]
        return self.post_audio(tts_speech, tts_text)

    def inference_instruct(self, tts_text, spk_id, instruct_text):  # 情感控制合成
        model_input = self.frontend.frontend_instruct(tts_text, spk_id, instruct_text)
        tts_speech = self.model.inference(**model_input)["tts_speech"]
        return self.post_audio(tts_speech, tts_text)

    def post_audio(self, tts_speech, tts_text):
        audio = tts_speech.numpy().flatten()
        audio = audio[2000:]
        audio_sample_rate = 22050

        # 增加静音隔断
        zero_wav = np.zeros(int(audio_sample_rate * 0.2), dtype=np.float32)

        # 字幕时间
        dur = len(audio) / audio_sample_rate
        num_wd = len(tts_text)
        step = dur / num_wd
        datas = [[wd, i * step, (i + 1) * step] for i, wd in enumerate(tts_text)]
        return audio, audio_sample_rate, datas


class CosyVoiceModel:
    def __init__(self, model_dir, fp16, use_jit, use_onnx, device):
        self.model_dir = model_dir
        self.fp16 = fp16
        self.use_jit = use_jit
        self.use_onnx = use_onnx
        self.device = device

        llm_model, flow_model, hift_model = create_model()

        self.llm = llm_model
        self.flow = flow_model
        self.hift = hift_model

    def load(self, llm_model, flow_model, hift_model):  # 加载权重
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

        if self.fp16:  # 转为半精度
            self.llm.half()

        if self.use_jit:
            self.load_jit()
        if self.use_onnx:
            self.load_onnx()

    def load_jit(self):  # 加载jit模型
        llm_text_encoder = torch.jit.load(f'{self.model_dir}/llm.text_encoder.fp16.zip', map_location=self.device)
        llm_llm = torch.jit.load(f'{self.model_dir}/llm.llm.fp16.zip', map_location=self.device)
        flow_encoder = torch.jit.load(f'{self.model_dir}/flow.encoder.fp32.zip', map_location=self.device)

        self.llm.llm = llm_llm
        self.llm.text_encoder = llm_text_encoder
        self.flow.encoder = flow_encoder

    def load_onnx(self):  # 加载onnx模型
        flow_decoder_estimator_model = f'{self.model_dir}/flow.decoder.estimator.fp32.onnx'
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        del self.flow.decoder.estimator
        self.flow.decoder.estimator = onnxruntime.InferenceSession(flow_decoder_estimator_model, sess_options=option,
                                                                   providers=providers)

    # 模型推断
    def inference(self, text, text_len, flow_embedding, llm_embedding=torch.zeros(0, 192),
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32), prompt_text_len=torch.zeros(1, dtype=torch.int32),
                  llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                  llm_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                  flow_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32)):

        if self.fp16:
            llm_embedding = llm_embedding.half()

        tts_speech_token = self.llm.inference(text=text.to(self.device),
                                              text_len=text_len.to(self.device),
                                              prompt_text=prompt_text.to(self.device),
                                              prompt_text_len=prompt_text_len.to(self.device),
                                              prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                              prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
                                              embedding=llm_embedding.to(self.device),
                                              beam_size=1,
                                              sampling=25,
                                              max_token_text_ratio=30,
                                              min_token_text_ratio=1)
        tts_mel = self.flow.inference(token=tts_speech_token,
                                      token_len=torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(
                                          self.device),
                                      prompt_token=flow_prompt_speech_token.to(self.device),
                                      prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                      prompt_feat=prompt_speech_feat.to(self.device),
                                      prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                      embedding=flow_embedding.to(self.device))
        tts_speech = self.hift.inference(mel=tts_mel).cpu()
        torch.cuda.empty_cache()
        return {'tts_speech': tts_speech}


class CosyVoiceFrontEnd:
    def __init__(self, campplus_model, speech_tokenizer_model, spk2info, instruct=False, allowed_special='all'):
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, num_languages=100, language='en',
                                                         task='transcribe')
        self.feat_extractor = mel_spectrogram
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option,
                                                             providers=["CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option,
                                                                     providers=[
                                                                         "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"])
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)

        self.instruct = instruct
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()

    def frontend_instruct(self, tts_text, spk_id, instruct_text):  # 情感控制合成输入
        model_input = self.frontend_sft(tts_text, spk_id)
        del model_input['llm_embedding']  # 因为是训练好的音色，所以不支持克隆
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        # 将instruct设置为prompt_text
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

    def frontend_sft(self, tts_text, spk_id):  # 官方音色合成输入
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)

        embedding = self.spk2info[spk_id]['embedding']

        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding,
                       'flow_embedding': embedding}
        return model_input

    def clone_spk(self, prompt_text, prompt_speech_16k):  # 提取prompt输入信息
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        embedding = self._extract_spk_embedding(prompt_speech_16k)

        model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def clone_text(self, tts_text, model_input_base):  # 获取合成文本token信息
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        model_input = copy.copy(model_input_base)
        model_input['text'] = tts_text_token
        model_input['text_len'] = tts_text_token_len
        return model_input

    def _extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len

    def _extract_speech_token(self, speech):
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None, {
            self.speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
            self.speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[
            0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None, {
            self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256,
                                          win_size=1024, fmin=0, fmax=8000, center=False).squeeze(dim=0).transpose(0,
                                                                                                                   1).to(
            self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len


def create_model():
    # llm model
    text_encoder_input_size = 512
    text_encoder = ConformerEncoder(input_size=text_encoder_input_size, output_size=1024, attention_heads=16,
                                    linear_units=4096, num_blocks=6,
                                    dropout_rate=0.1, positional_dropout_rate=0.1,
                                    attention_dropout_rate=0.0, normalize_before=True,
                                    input_layer=LinearNoSubsampling,
                                    pos_enc_layer_type=EspnetRelPositionalEncoding,
                                    selfattention_layer_type=RelPositionMultiHeadedAttention, use_cnn_module=False,
                                    macaron_style=False, use_dynamic_chunk=False,
                                    use_dynamic_left_chunk=False, static_chunk_size=1)
    llm_encoder = TransformerEncoder(input_size=1024, output_size=1024, attention_heads=16, linear_units=4096,
                                     num_blocks=14, dropout_rate=0.1, positional_dropout_rate=0.1,
                                     attention_dropout_rate=0.0, input_layer=LegacyLinearNoSubsampling,
                                     pos_enc_layer_type=EspnetRelPositionalEncoding,
                                     selfattention_layer_type=RelPositionMultiHeadedAttention,
                                     static_chunk_size=1)

    llm = TransformerLM(text_encoder_input_size=text_encoder_input_size, llm_input_size=1024,
                        llm_output_size=1024, text_token_size=51866,
                        speech_token_size=4096, length_normalized_loss=True, lsm_weight=0.0, spk_embed_dim=192,
                        text_encoder=text_encoder, llm=llm_encoder)

    flow_encoder = ConformerEncoder(output_size=512, attention_heads=8, linear_units=2048, num_blocks=6,
                                    dropout_rate=0.1, positional_dropout_rate=0.1,
                                    attention_dropout_rate=0.1, normalize_before=True,
                                    input_layer=LinearNoSubsampling,
                                    pos_enc_layer_type=EspnetRelPositionalEncoding,
                                    selfattention_layer_type=RelPositionMultiHeadedAttention, input_size=512,
                                    use_cnn_module=False,
                                    macaron_style=False)

    estimator_decoder = ConditionalDecoder(in_channels=320, out_channels=80, channels=[256, 256], dropout=0,
                                           attention_head_dim=64, n_blocks=4, num_mid_blocks=12,
                                           num_heads=8, act_fn='gelu')
    cfm_params = DictConfig(
        content={'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine', 'training_cfg_rate': 0.2,
                 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'})

    flow = MaskedDiffWithXvec(input_size=512, output_size=80, spk_embed_dim=192, output_type='mel',
                              vocab_size=4096, input_frame_rate=50,
                              only_mask_loss=True, encoder=flow_encoder,
                              length_regulator=InterpolateRegulator(channels=80, sampling_ratios=[1, 1, 1, 1]),
                              decoder=ConditionalCFM(in_channels=240, n_spks=1, spk_emb_dim=80,
                                                     cfm_params=cfm_params, estimator=estimator_decoder))
    # hift model
    hift = HiFTGenerator(in_channels=80, base_channels=512, nb_harmonics=8, sampling_rate=22050, nsf_alpha=0.1,
                         nsf_sigma=0.003, nsf_voiced_threshold=10,
                         upsample_rates=[8, 8], upsample_kernel_sizes=[16, 16],
                         istft_params={'n_fft': 16, 'hop_len': 4}, resblock_kernel_sizes=[3, 7, 11],
                         resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                         source_resblock_kernel_sizes=[7, 11], source_resblock_dilation_sizes=
                         [[1, 3, 5], [1, 3, 5]], lrelu_slope=0.1, audio_limit=0.99,
                         f0_predictor=ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512))

    return llm, flow, hift


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)
def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    prompt_sr = 16000
    target_sr = 22050
    model_dir = COSYVOICE_MODEL_DIR
    cosyvoice = CosyVoiceInfer(model_dir, use_jit=True, fp16=True, device=device)  # 加载模型

    # 加载合成文本
    tts_text_list = []
    with open('../resources/tn.txt', 'r') as f:
        for line in f.readlines():
            tts_text_list.append(line.strip())

    tts_front = TTS_Front()

    cost_time = []
    for index, tts_text in enumerate(tqdm(tts_text_list)):
        start_time = time.time()
        audios = []

        texts = tts_front.process(tts_text, split=True)
        for tts_text_str in texts:
            # result = cosyvoice.inference_text(tts_text)
            audio, audio_sample_rate, datas = cosyvoice.inference_sft(tts_text_str, spk_id='baoyi')
            # result = cosyvoice.inference_instruct(tts_text, spk_id='weiwei', instruct_text='You are a female speaker with a slightly faster speaking pace, normal tone, and stable emotions.')

            audios.append(audio)
        audios = np.concatenate(audios)
        file_name = f'{time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))}_{index}'
        sf.write(f'../resources/{file_name}.wav', audios, target_sr, subtype='PCM_16')

        format_time = time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))
        audio_time = len(audios) / target_sr
        spend_time = time.time() - start_time
        cost_time.append(spend_time)
        print(f'RTF = {spend_time / audio_time}')