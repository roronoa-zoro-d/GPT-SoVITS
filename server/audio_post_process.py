import sys
import os
import soundfile as sf
import librosa
import numpy as np
import resampy
import pyrubberband as pyrb


def audio_post_process(audio, in_fs, out_fs, volume_ratio, speed_ratio, pitch_ratio):
    print(f'audio post: audio.shape {audio.shape}, type {type(audio)}, dtype {audio.dtype}')
    if audio.dtype == np.int16:
        audio = (audio/32768.0).astype(np.float32)
    elif audio.dtype == np.float16:
        audio = audio.astype(np.float32)
        
    if in_fs != out_fs:
        audio = resampy.resample(audio, in_fs, out_fs)
    if speed_ratio != 1.0:
        audio = pyrb.time_stretch(audio, out_fs, speed_ratio)
        
    if pitch_ratio != 1.0:
        audio = pyrb.pitch_shift(audio, out_fs, pitch_ratio)
    
    if volume_ratio != 1.0:
        audio = audio * volume_ratio
        max_audio=np.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:
            audio/=max_audio
    
    audio = (audio * 32768.0).astype(np.int16)
    return audio
    
    
    
if __name__ == '__main__':

    wav_path = 'out_wav_admin_v2.wav'
    
    speech, fs = sf.read(wav_path, dtype='int16')
    
    speech = audio_post_process(speech, fs, 16000, 1.0, 1.0, 1.0)
    sf.write('out_clone_post.wav', speech, 16000)
    
    print(f'audio shape {speech.shape[0]}')
    
    

