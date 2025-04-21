import sys
import os
import logging

import redis
from rediscluster import RedisCluster
from joblib import dump, load
from pickle import dumps, loads
import base64

from yto_utils import init_file_logger, read_scp, download_file

from conf import GPT_ROOT_DIR, SERVER_ROOT_DIR, GPT_SOVITS_MODEL_DIR
sys.path.append(SERVER_ROOT_DIR)
logger = init_file_logger('speaker_embedding', level=logging.DEBUG)


# Redis 集群节点配置
startup_nodes = [
    {"host": "10.6.112.58", "port": "6429"},
    {"host": "10.6.112.58", "port": "6430"},
    {"host": "10.6.112.62", "port": "6429"},
    {"host": "10.6.112.62", "port": "6430"},
    {"host": "10.6.112.66", "port": "6429"},
    {"host": "10.6.112.66", "port": "6430"}
]

class SpeakerEmbeddings(object):
    def __init__(self, emb_dir=f'{SERVER_ROOT_DIR}/spk_embs', ):
        self.spk_emb_dir = f'{emb_dir}/spk_embs'
        self.spk_scp_file = f'{emb_dir}/spk_emb.scp'
        
        if os.path.exists(self.spk_emb_dir) is False:
            os.makedirs(self.spk_emb_dir, exist_ok=True)
        
        self.spk_names = []
        self.spk2path = {}
        self.spk2data = {}
        self.load_spk_embs()
    
    def load_spk_embs(self):
        if not os.path.exists(self.spk_scp_file):
            logger.warning(f'can not load spk_emb, file not  exists: {self.spk_scp_file}')
            return
        
        spk_names, spk2path = read_scp(self.spk_scp_file)
        for spk_name in spk_names:
            try:
                # logger.debug(f'load spk emb: {spk_name}')
                emb_path = spk2path[spk_name]
                if os.path.exists(emb_path) is False:
                    logger.warning(f'spk_emb {spk_name} not exists: {emb_path}')
                    continue
                data = load(emb_path)
                self.spk2data[spk_name] = data
                self.spk2path[spk_name] = emb_path
                self.spk_names.append(spk_name)
            except Exception as e:
                logger.error(f'load spk_emb {spk_name} error: {e}')
        logger.info(f'load spks total {len(self.spk2data)} spk, [{spk_names}]')

    
    
    def save_spk_emb(self, spk_name, spk_data):

        # 强制重写
        emb_path = f'{self.spk_emb_dir}/{spk_name}.spk_emb'
        dump(spk_data, emb_path)
        logger.info(f'save spk_emb {spk_name} to {emb_path}')
        
        with open(self.spk_scp_file, 'a') as f:
            f.write(f'{spk_name} {emb_path}\n')
        
        self.spk_names.append(spk_name)
        self.spk2data[spk_name] = spk_data
        self.spk2path[spk_name] = emb_path
        return emb_path
    
    def get_spk_names(self,):
        return self.spk_names
    

    def get_spk_emb(self, spk_name):
        if spk_name not in self.spk2data:
            return None
        
        return self.spk2data[spk_name]



class SpeakerEmbeddingRedis(object):
    def __init__(self):
        try:
            if 'workspace' in os.path.dirname(__file__):
                # self.rds = redis.Redis(host='10.199.27.147', port=6379, db=0)
                # 创建 Redis 集群连接
                self.rds = RedisCluster(startup_nodes=startup_nodes, decode_responses=True, socket_timeout=5)
                print("Connected to Redis Cluster")
            
            else:
                self.rds = redis.Redis(host='localhost', port=6379, db=0)
                print("Connected to localhost")
        except Exception as e:
                print(f"Failed to connect to Redis : {e}")
                exit(1)
    def get_spk_data(self, spk_id):
        if self.rds.exists(spk_id) is False:
            logger.error(f'no speaker id in redis : {spk_id}')
            return None
        serialized_data = self.rds.get(spk_id)
        if serialized_data is None:
            logger.error(f'error serialized_data is None in redis : {spk_id}')
            return None
        decoded_data = base64.b64decode(serialized_data)
        spk_data = loads(decoded_data)
        if 'norm_text' not in spk_data or 'speech' not in spk_data:
            logger.error(f'no norm_text or speech key in spk_data: {spk_id}')
            return None            
        return spk_data
    
    def save_spk_data(self, spk_id, spk_data):
        serialized_data = dumps(spk_data)
        encoded_data = base64.b64encode(serialized_data).decode('utf-8')
        self.rds.set(spk_id, encoded_data)
        