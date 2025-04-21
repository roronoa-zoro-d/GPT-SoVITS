import requests

def get_spk():
    # 服务端运行的 URL
    url = "http://localhost:8000/gpt_sovits/speakers"

    # 发送 GET 请求
    response = requests.get(url)

    # 确保请求成功
    if response.status_code == 200:
        # 解析响应的 JSON 数据
        speakers = response.json()
        # 打印说话人列表
        print("Speakers:", speakers["speakers"])
    else:
        print(f"Failed to get speakers, status code: {response.status_code}")
        
    
# get_spk()



import asyncio
import json
import websockets
import logging
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO)


async def test_connect_to_server(uri):
    # uri = "ws://127.0.0.1:8000/test_connect"
    async with websockets.connect(uri) as websocket:
        print("Connected to server")
        
        while True:
            message = await websocket.recv()
            print(f"Received from server: {message}")
            if message == "finish":
                break

async def connect_to_server(uri):
    async with websockets.connect(uri) as websocket:
        print("Connected to server")
        await websocket.send(json.dumps({"text": "网点的盈利状况，是由收入提升和成本控制两方面决定的，圆通网点要想赚钱，主要是做好“三升四降”。今天天气怎么样。今天天气非常好。你今天去哪里玩。", "spk": "aa"}))
        sample_rate = 16000
        audio_data = np.array([], dtype=np.int16)
        while True:
            message = await websocket.recv()
            # print("Received from server:", message)  # 打印接收到的消息

            # 判断是二进制数据还是文本数据
            if isinstance(message, bytes):
                audio_chunk = np.frombuffer(message, dtype=np.int16)
                audio_data = np.concatenate((audio_data, audio_chunk))
            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                    if 'is_finish' in data and data['is_finish'] == 'true':
                            print("Audio recording finished.")
                            sf.write("out.wav", audio_data, sample_rate)
                            break
                except json.JSONDecodeError:
                    print(f"Received text: {message}")
            else:
                raise ValueError("Unsupported message type received.")

        await websocket.close()

async def main():
    uri = "ws://localhost:8000/gpt_sovits/gen_audio"
    await connect_to_server(uri)
    # await test_connect_to_server(uri)

if __name__ == "__main__":
    asyncio.run(main())