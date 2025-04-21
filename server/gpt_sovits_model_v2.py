import os
import re
import regex
import logging
import sys

import torch
import soundfile as sf

import numpy as np
import librosa

# from conf import GPT_ROOT_DIR, SERVER_ROOT_DIR, GPT_SOVITS_MODEL_DIR
from conf import GPT_V2_ROOT_DIR, SERVER_ROOT_DIR, GPT_SOVITS_MODEL_DIR




sys.path.append(GPT_V2_ROOT_DIR)

import LangSegment
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from transformers import AutoModelForMaskedLM, AutoTokenizer
from module.mel_processing import spectrogram_torch
from text.cleaner import clean_text
from text import cleaned_text_to_sequence

sys.path.append(SERVER_ROOT_DIR)
from yto_utils import init_file_logger
from yto_utils import speech_rtf_decorator, text_rtf_decorator
from punctuation_process import punctuation_process
from speaker_embedding import SpeakerEmbeddings
# from gpt_sovits_cleaner import GPTCleaner

logger = init_file_logger('gpt_sovits_model_v2', level=logging.INFO, propagate=True)





# 暂时只支持中英，先不考虑日韩。
LangSegment.setfilters(["zh","en",])
class MultiLanguageProcess(object):
    def __init__(self, language='zh'):
        self.language = language
        # self.gpt_cleaner = GPTCleaner()

    def process(self, text):
        text = punctuation_process(text, self.language)
        
        datas = []
        txt_lang_segs = LangSegment.getTexts(text)
        for txt_lang_seg in txt_lang_segs:
            lang = txt_lang_seg['lang']
            txt = txt_lang_seg['text']
            # phones, word2ph, norm_text = self.gpt_cleaner.clean_text(txt, lang)
            phones, word2ph, norm_text = clean_text(txt, lang, version='v2')
            phone_ids = cleaned_text_to_sequence(phones, version='v2')
            data = {}
            data['lang'] = lang
            data['phones'] = phones
            data['phone_ids'] = phone_ids
            data['text'] = txt
            data['norm_text'] = norm_text
            data['word2ph'] = word2ph
            datas.append(data)
            
            logger.debug(f'language: {lang} , text: {txt}')
            logger.debug(f'norm_text: {norm_text}')
            logger.debug(f'phones len {len(phones)}: {phones}')
        
        return datas


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")



class zoro_gpt_sovits(object):
    def __init__(self,device=None):
        
        # self.pretrain_gpt = f"{GPT_SOVITS_MODEL_DIR}/pretrained_models_v2/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        # self.pretrained_sovits = f"{GPT_SOVITS_MODEL_DIR}/pretrained_models_v2/gsv-v2final-pretrained/s2G2333k.pth"
        # self.model_name = 'gpt_sovits_pretrain_model_v2'
        
        # self.pretrain_gpt = f"{GPT_SOVITS_MODEL_DIR}/GPT_weights_v2/exp1-e10.ckpt"
        # self.pretrained_sovits = f"{GPT_SOVITS_MODEL_DIR}/SoVITS_weights_v2/exp1_e8_s88.pth"
        # self.model_name = 'gpt_sovits_pretrain_model_v2_finetune_exp1_g10_s88'
        
        # self.pretrain_gpt = f"{GPT_SOVITS_MODEL_DIR}/GPT_weights_v2/exp2-e15.ckpt"
        # self.pretrained_sovits = f"{GPT_SOVITS_MODEL_DIR}/SoVITS_weights_v2/exp2_e8_s56.pth"
        # self.model_name = 'gpt_sovits_pretrain_model_v2_finetune_exp2_g15_s56'
        
        # 宝怡维维，火山克隆生成数据进行微调
        self.pretrain_gpt = f"{GPT_SOVITS_MODEL_DIR}/GPT_weights_v2/exp3-e5.ckpt"
        self.pretrained_sovits = f"{GPT_SOVITS_MODEL_DIR}/SoVITS_weights_v2/exp3_e4_s108.pth"
        self.model_name = 'gpt_sovits_pretrain_model_v2_finetune_exp3_g5_s108'
        
        model_path = {}
        model_path['gpt_path'] = self.pretrain_gpt
        model_path['sovits_path'] = self.pretrained_sovits
        model_path['cnhubert_base_path'] = f"{GPT_SOVITS_MODEL_DIR}/pretrained_models_v2/chinese-hubert-base"
        model_path['bert_path'] = f"{GPT_SOVITS_MODEL_DIR}/pretrained_models_v2/chinese-roberta-wwm-ext-large"
        self.model_path = model_path
        
        self.gpt_models = []
        self.gpt_models.append([os.path.basename(model_path['gpt_path']), model_path['gpt_path']])
        self.sovits_models = []
        self.sovits_models.append([os.path.basename(model_path['sovits_path']), model_path['sovits_path']])

        self.is_half = True
        self.dtype_np = np.float16 if self.is_half == True else np.float32
        self.dtype_torch = torch.float16 if self.is_half == True else torch.float32
        self.device = torch.device("cpu")
        if device is not None:
            self.device = device
        logger.info(f"device: {self.device}")
        
        self.multi_language = MultiLanguageProcess(language='zh')
        
        #默认合成音色
        # self.tts_spk_embs = SpeakerEmbeddings(emb_dir=f'{SERVER_ROOT_DIR}/spk_embs/GPT_SoVits_v2/')
        # self.tts_spk_embs = SpeakerEmbeddings(emb_dir=f'{SERVER_ROOT_DIR}/spk_embs/GPT_SoVits_v2_finetune_finetune_exp1_g10_s88/')
        # self.tts_spk_embs = SpeakerEmbeddings(emb_dir=f'{SERVER_ROOT_DIR}/spk_embs/GPT_SoVits_v2_finetune_finetune_exp2_g15_s56/')
        self.tts_spk_embs = SpeakerEmbeddings(emb_dir=f'{SERVER_ROOT_DIR}/spk_embs/gpt_sovits_pretrain_model_v2_finetune_exp3_g5_s108/')
        self.tts_spk_embs.load_spk_embs()
        
        self.init_model()
        print('gpt_sovits load model success....')
        
    def get_spk_names(self,):
        spk_names = set(self.tts_spk_embs.get_spk_names())
        logger.info(f'return spk_names: {spk_names}')
        return spk_names

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language=language.replace("all_","")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)#.to(dtype)
        else:
            bert = torch.zeros((1024, len(phones)),dtype=self.dtype_torch,).to(self.device)

        return bert

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_phones_and_bert(self, text, language='zh'):
        phones_list = []
        bert_list = []
        norm_text_list = []
        datas = self.multi_language.process(text)
        logger.debug(f'multi-language process: {datas}')
        for data in datas:
            bert = self.get_bert_inf(data['phone_ids'], data['word2ph'], data['norm_text'], data['lang'])
            phones_list.append(data['phone_ids'])
            norm_text_list.append(data['norm_text'])
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

        return phones,bert.to(self.dtype_torch),norm_text
    
    def load_sovits_weights(self, sovits_path=""):
        if sovits_path != "" and os.path.exists(sovits_path):
            self.model_path['sovits_path'] = sovits_path
        
        logger.info(f'load sovits model: {self.model_path["sovits_path"]}')
        dict_s2 = torch.load(self.model_path['sovits_path'], map_location="cpu")
        
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"
        self.hps = hps
        
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        if ("pretrained" not in sovits_path):
            del vq_model.enc_q
        if self.is_half == True:
            vq_model = vq_model.half().to(self.device)
        else:
            vq_model = vq_model.to(self.device)
        vq_model.eval()
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        self.vq_model = vq_model


    def load_gpt_weights(self, gpt_path=""):
        
        if gpt_path != "" and os.path.exists(gpt_path):
            self.model_path['gpt_path'] = gpt_path
        
        logger.info(f'load gpt model: {self.model_path["gpt_path"]}')
        self.hz = 50
        dict_s1 = torch.load(self.model_path['gpt_path'], map_location="cpu")
        self.gpt_config = dict_s1["config"]
        self.max_sec = self.gpt_config["data"]["max_sec"]
        logger.debug(f't2s_model config: {self.gpt_config}')
        t2s_model = Text2SemanticLightningModule(self.gpt_config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half == True:
            t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(self.device)
        t2s_model.eval()
        self.t2s_model = t2s_model
        
        total = sum([param.nelement() for param in self.t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))



    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path['bert_path'])
        
        bert_model = AutoModelForMaskedLM.from_pretrained(self.model_path['bert_path'])
        if self.is_half == True:
            bert_model = bert_model.half().to(self.device)
        else:
            bert_model = bert_model.to(self.device)
        self.bert_model = bert_model
        
        cnhubert.cnhubert_base_path = self.model_path['cnhubert_base_path']
        ssl_model = cnhubert.get_model()
        if self.is_half == True:
            ssl_model = ssl_model.half().to(self.device)
        else:
            ssl_model = ssl_model.to(self.device)
        self.ssl_model = ssl_model
        
        self.load_sovits_weights()
        self.load_gpt_weights()

    def get_spepc(self, audio:np.ndarray, fs=16000):
        audio = audio.astype(np.float32)
        if fs != self.hps.data.sampling_rate:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=self.hps.data.sampling_rate)
        audio = torch.FloatTensor(audio)
        maxx=audio.abs().max()
        if(maxx>1):audio/=min(2,maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        logger.debug(f'audio shape: {audio_norm.shape}, type {type(audio_norm)}, dtype {audio_norm.dtype}')
        spec = spectrogram_torch(audio_norm, self.hps.data.filter_length,
            self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length, center=False,)
        
        if self.is_half == True:
            spec = spec.to(torch.float16).to(self.device)
        else:
            spec = spec.to(torch.float32).to(self.device)
            
        logger.debug(f'---debug---  get_spec audio_norm shape {audio_norm.shape} sum {torch.sum(torch.abs(audio_norm))}')
        logger.debug(f'---debug---  get_spec filter_length {self.hps.data.filter_length}, sampling_rate {self.hps.data.sampling_rate}, hop_length {self.hps.data.hop_length}, win_length {self.hps.data.win_length}')
        
        return spec
    
    def generate_spk_emb(self, wav16k:np.ndarray, fs=16000, text=None, language=None):
        return self.generate_spk_emb_admin(wav16k, wav16k, fs=16000, text=text, language=language)
    

    @speech_rtf_decorator(1)
    def generate_spk_emb_admin(self, wav32k:np.ndarray, wav16k, fs=32000, text=None, language=None):
        # 生成mel的音频时32k， 生成prompt的音频时16k， 都是 float32 类型
        # if (speech.shape[0] > 160000 or speech.shape[0] < 48000):
        #     raise OSError(("参考音频在3~10秒范围外，请更换！"))
        
        spec = self.get_spepc(wav32k, fs)
        
        wav16k = wav32k
        if fs != 16000:
            wav16k = librosa.resample(wav32k, orig_sr=fs, target_sr=16000)
            
        logger.debug(f'---debug--- ori-wav16k shape {wav16k.shape}, sum = {np.sum(wav16k)}')
        speech = torch.from_numpy(wav16k)
        
        zero_wav = np.zeros(int(self.hps.data.sampling_rate * 0.3),dtype=self.dtype_np,)
        zero_wav = torch.from_numpy(zero_wav)
        
        if self.is_half:
            speech = speech.half()
            zero_wav = zero_wav.half()
        wav16k = torch.cat([speech, zero_wav]).to(self.device)
        with torch.no_grad():
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose( 1, 2)  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)
            logger.info(f'---debug--- ref-prompt shape {prompt.shape} sum = {torch.sum(prompt)}')

        cpu_device = torch.device('cpu')
    
        spk_info = {}
        spk_info['speech'] = speech.to(cpu_device)
        spk_info['ref_free'] = True
        spk_info['norm_text'] = None
        
        spk_data = {}
        spk_data['spec'] = spec.to(cpu_device)
        spk_data['prompt'] = prompt.to(cpu_device)
        spk_data['phones'] = None
        spk_data['bert'] = None
        
        logger.debug(f'---debug ref-spk --- text: {text}')
        logger.debug(f'---debug ref-spk --- wav16k shape {wav16k.shape}, sum = {torch.sum(wav16k)}')
        logger.debug(f'---debug ref-spk --- ref-prompt shape {prompt.shape} sum = {torch.sum(prompt)}')

        if text is not None:
            phones,bert,norm_text=self.get_phones_and_bert(text, language)
            spk_data['phones'] = phones
            spk_data['bert'] = bert.to(cpu_device)
            spk_info['norm_text'] = norm_text
            spk_info['ref_free'] = False
            
            logger.debug(f'---debug ref-spk --- phones {phones}')
            logger.debug(f'---debug ref-spk --- norm_text1 {norm_text}')
            logger.debug(f'---debug ref-spk --- bert  shape {bert.shape} sum {bert.sum()}') 
            logger.debug(f'---debug ref-spk --- spec type {type(spec)}, shape {spec.shape} sum {torch.sum(torch.abs(spec))}')
            # logger.debug(f'---debug ref-spk --- spec : {spec[0,0:10,0:10]}')

            
        spk_info[self.model_name] = spk_data
        
        return spk_info


    @text_rtf_decorator(1)
    def infer(self, text, language, ref_spk, top_k=1, top_p=1, temperature=1):
        logger.info(f'infer text: {text} ref_spk-text {ref_spk["norm_text"]} ')
        phones,bert,norm_text=self.get_phones_and_bert(text, language)
        
        if len(phones) == 0:
            zero_wav = np.zeros(int(self.hps.data.sampling_rate * 0.1),dtype=self.dtype_np,)
            return  (zero_wav*32768).astype(np.int16), self.hps.data.sampling_rate, []
        
        ref_spk_data = ref_spk[self.model_name]
        
        if ref_spk['ref_free'] is False:
            all_phoneme_ids = torch.LongTensor(ref_spk_data['phones'] + phones).to(self.device).unsqueeze(0)
            prompt = ref_spk_data['prompt'].to(self.device)
            bert = torch.cat([ref_spk_data['bert'].to(self.device), bert], 1)
        else:
            all_phoneme_ids = torch.LongTensor(phones).to(self.device).unsqueeze(0)
            prompt = None
            
        
        bert = bert.to(self.device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

        logger.info(f'infer text: {text}')
        with torch.no_grad():
            for i in range(3):
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
                if pred_semantic.shape[2] < 1000:
                    break
                
                top_k += 5
                logger.warning(f'predA_semantic-shape2= {pred_semantic.shape[2]}, do again')
        
        ref_phone_len = len(ref_spk_data['phones'])
        logger.info(f'--count-- pred_semantic-len {pred_semantic.shape[2]} infer-phones-len {len(phones)} ref-text-len {ref_phone_len} ')

        
        
        logger.info(f'-----debug----- text: {text}')
        logger.info(f'-----debug----- all_phoneme_ids: sum-{torch.sum(all_phoneme_ids)}')
        logger.info(f'-----debug----- all_phoneme_len: {all_phoneme_len}')
        logger.info(f'-----debug----- ref-prompt shape {prompt.shape} sum = {torch.sum(prompt)}')
        logger.debug(f'----debug----- ref_spk bert : shape {bert.shape} sum {torch.sum(torch.abs(bert))} subsum-{torch.sum(torch.abs(bert[0,0:10,0:10]))}') 
        logger.debug(f'----debug-----  pred_semantic type {type(pred_semantic)}, shape {pred_semantic.shape} sum {torch.sum(torch.abs(pred_semantic))}')
        logger.debug(f'----debug-----  top_k = {top_k}, top_p = {top_p}, temperature = {temperature} early_stop_num = {self.hz * self.max_sec}' )        
        refers = [ref_spk_data['spec'].to(self.device)]
        
        audio = self.vq_model.decode(pred_semantic, torch.LongTensor(phones).to(self.device).unsqueeze(0), [ref_spk_data['spec'].to(self.device)],speed=1)
        audio = audio.detach().cpu().numpy()[0, 0]
        logger.debug(f'----debug----- audio shape {audio.shape} sum {np.sum(np.abs(audio))}')
        logger.debug(f'----debug----- num-refer: {len(refers)}, shape {[x.shape for x in refers]}, sum-{[torch.sum(x[0,0:10,0:10]) for x in refers]}')
        logger.debug(f'----debug----- phone: {phones}')
        max_audio=np.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:
            audio/=max_audio
        
        
        sampling_rate = self.hps.data.sampling_rate
        dur = len(audio)/sampling_rate
        num_wd = len(norm_text)
        step = dur/num_wd
        datas = []
        for i, wd in enumerate(norm_text):
            datas.append([wd,i*step, (i+1)*step])
        
        
        zero_wav = np.zeros(int(self.hps.data.sampling_rate * 0.3),dtype=self.dtype_np,)
        final_audio = np.concatenate([audio, zero_wav], 0)
        logger.debug(f'generate final audio shape {final_audio.shape} fs {sampling_rate}, type {type(final_audio)}')
        
        # return final_audio, self.hps.data.sampling_rate
        return audio, sampling_rate, datas


    def clone_infer(self, text, language, ref_spk, top_k=5, top_p=1, temperature=1):
        return self.infer(text, language, ref_spk, top_k, top_p, temperature)
    
    def tts_infer(self, text, language, tts_spk,  top_k=5, top_p=1, temperature=1):
        spk_data = self.tts_spk_embs.get_spk_emb(tts_spk)
        if spk_data is None:
            zero_wav = np.zeros(int(self.hps.data.sampling_rate * 0.3),dtype=self.dtype_np,)
            logger.error(f'tts_spk {tts_spk} not found')
            return zero_wav, self.hps.data.sampling_rate
        
        return self.infer(text, language, spk_data, top_k, top_p, temperature)
    def get_models(self):
        self.gpt_models = {}
        self.gpt_models[os.path.basename(self.pretrain_gpt)] = self.pretrain_gpt
        gpt_weight_dir = f'{GPT_SOVITS_MODEL_DIR}/GPT_weights'
        if 'gpt_sovits_pretrain_model_v2' in self.model_name:
            gpt_weight_dir = f'{GPT_SOVITS_MODEL_DIR}/GPT_weights_v2'
        for name in os.listdir(gpt_weight_dir):
            if name.endswith(".ckpt"):
                self.gpt_models[name] = f'{gpt_weight_dir}/{name}'

        self.sovits_models = {}
        self.sovits_models[os.path.basename(self.pretrained_sovits)] = self.pretrained_sovits
        sovits_weight_dir = f'{GPT_SOVITS_MODEL_DIR}/SoVITS_weights'
        if 'gpt_sovits_pretrain_model_v2' in self.model_name:
            sovits_weight_dir = f'{GPT_SOVITS_MODEL_DIR}/SoVITS_weights_v2'
        for name in os.listdir(sovits_weight_dir):
            if name.endswith(".pth"):
                self.sovits_models[name] = f'{sovits_weight_dir}/{name}'
        
        # gpt_model_names = sorted(self.gpt_models.keys())
        # sovits_model_names = sorted(self.sovits_models.keys())
        # return gpt_model_names, sovits_model_names

        return self.gpt_models, self.sovits_models



if __name__ == "__main__":
    device = torch.device('cuda:0')
    model = zoro_gpt_sovits(device=device)


    ref_wav_path = '/home/zhangjiayuan/program/audio/tts/ref_wav/naijie_split01.wav'
    ref_txt = '网点的盈利状况是由收入提升和成本控制两方面决定的。'
    language = 'zh'
    
    ref_speech, sr = librosa.load(ref_wav_path, sr=16000)
    use_ref = True
    
    spk_info = model.generate_spk_emb(ref_speech, text=ref_txt if use_ref else None, language=language)
    
    text = '你好，请问一下， 华兴路怎么走。'
    audio, sample_rate, stamp = model.infer(text, language, spk_info, top_k=5, top_p=1, temperature=1)
    print(f'generate audio: shape {audio.shape}, type {type(audio)}, dtype {audio.dtype}')
    
    
    
    sf.write('out.wav', audio, sample_rate)