'''
[20240603]
'''
import sys
import os
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))
from typing import Generator
import io
from io import BytesIO
import subprocess
import numpy as np
import soundfile as sf
# import pyaudio
import wave
# import GPT_SoVITS
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config


def pack_ogg(io_buffer:BytesIO, data:np.ndarray, rate:int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer


def pack_aac(io_buffer:BytesIO, data:np.ndarray, rate:int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()




'''global'''
default_prompt_lang = "zh"
# default_ref_audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data/zl.000.wav"))
# default_prompt_text = "为流通而造的船，遇到港口也会停泊，所以璃月是一切财富沉淀的地方。"

# default_ref_audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data/fnn.000.wav"))
# default_prompt_text = "为你献上！让世界热闹起来吧！"

default_ref_audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data/fnn.001.wav"))
default_prompt_text = "茶会是淑女的必修课，如果你想学习茶会礼仪的话，我可以教你哦。"


config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "GPT_SoVITS/configs/tts_infer.yaml")
)
tts_config = TTS_Config(config_path)
tts_pipeline = TTS(tts_config)


def tts_infer(
    text: str,
    text_lang: str = "zh",
    ref_audio_path: str = default_ref_audio_path,
    prompt_text: str = default_prompt_text,
    prompt_lang: str = default_prompt_lang,
    media_type: str = "wav",
    streaming_mode: bool = False,
):
    global tts_pipeline
    req = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "top_k": 5,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": "cut5",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": -1,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }
    
    if streaming_mode:
        req["return_fragment"] = True
    
    try:
        tts_generator = tts_pipeline.run(req)
        if streaming_mode:
            def streaming_generator(tts_generator:Generator, media_type:str):
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                for sr, chunk in tts_generator:
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
            return streaming_generator(tts_generator, media_type)
        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return audio_data
    except Exception as e:
        print({"message": "tts failed", "Exception": str(e)})


def test_tts_infer(streaming_mode = False):
    text = "《杀死一只知更鸟》是美国女作家哈珀·李发表于1960年的长篇小说。该小说讲述一个名叫汤姆·鲁滨逊的年轻人,被人诬告犯了强奸罪后,只是因为是一个黑人,辩护律师阿蒂克斯·芬奇尽管握有汤姆不是强奸犯的证据,都无法阻止陪审团给出汤姆有罪的结论。此一妄加之罪,导致汤姆死于乱枪之下。虽然故事题材涉及种族不平等与强暴等严肃议题,其文风仍温暖风趣。"
    text = "《杀死一只知更鸟》是美国女作家哈珀·李发表于1960年的长篇小说。该小说讲述一个名叫汤姆·鲁滨逊的年轻人,被人诬告犯了强奸罪后,只是因为是一个黑人,辩护律师阿蒂克斯·芬奇尽管握有汤姆不是强奸犯的证据,都无法阻止陪审团给出汤姆有罪的结论。"
    text_lang = "zh"
    ref_audio_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "./data/zl.000.wav")
    )
    prompt_text = "为流通而造的船，遇到港口也会停泊，所以璃月是一切财富沉淀的地方。"
    prompt_lang = "zh"
    import time
    start_time = time.time()
    audio_data = tts_infer(
        text = text,
        # text_lang = text_lang,
        # ref_audio_path = ref_audio_path, 
        # prompt_text = prompt_text,
        # prompt_lang = prompt_lang,
        streaming_mode = streaming_mode,
    )
    print(time.time() - start_time)

    output_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "output.wav")
    )
    if os.path.exists(output_path):
        os.remove(output_path)
    with open('output.wav', 'wb') as f:
        if isinstance(audio_data, Generator):
            for data in audio_data:
                f.write(data)
        else:
            f.write(audio_data)


if __name__ == "__main__":
    test_tts_infer(False)
    
    # test_tts_infer(True)

    pass


