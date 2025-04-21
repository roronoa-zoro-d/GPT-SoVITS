import redis
import uuid
import librosa
import numpy as np
from pickle import dumps, loads

from conf import SERVER_ROOT_DIR
from joblib import dump, load
from pickle import dumps, loads
import base64

rds = redis.Redis(host='127.0.0.1', port=6379, db=0)

# 获取redis配置
redis_config = rds.config_get('*')
print(redis_config['save'])

# 假设你要从 Redis 中获取一个键为 'my_key' 的值
key = 'naijie-694272b4-19af-4939-83f6-dab7d9a221f9'
# value = rds.set(key, 'hello world')


if rds.exists(key):
    serialized_data = rds.get(key)
    # print(f"Key '{key}' exists in Redis. value is : {value}")
    decoded_data = base64.b64decode(serialized_data)
    spk_data = loads(decoded_data)
    print(spk_data)
    if 'norm_text' not in spk_data or 'speech' not in spk_data:
        print('error')
else:
    print(f"Key '{key}' does not exist in Redis.")



