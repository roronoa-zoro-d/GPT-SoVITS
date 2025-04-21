import sys
import os
import re
import yaml
import logging

from conf import SERVER_ROOT_DIR
from conf import Punctuation, serial_number_map
from yto_utils import init_file_logger

logger = init_file_logger('cut_split', level=logging.DEBUG, propagate=False)


def split_by_punc(text, punc_str):
    punc_pattern = "(.*?)([" + re.escape(punc_str) + "])"
    split_text = []
    matches = list(re.finditer(punc_pattern, text))
    last_end = 0
    for match in matches:
            start, end = match.span()
            prefix,  punc = match.groups()
            text_seg = prefix + punc 
            if len(text_seg) > 0:
                split_text.append(text_seg)
            last_end = end
    
    text_seg = text[last_end:]
    if len(text_seg) > 0:
        split_text.append(text_seg)
    
    return split_text

def merge_by_len_range(texts:list, min_len, max_len):
    if len(texts) == 0:
        return []
    split_texts = []
    # split_texts.append(texts[0])
    
    text_prefix = ""
    for text in texts:
        if len(text_prefix) + len(text) < max_len:
            text_prefix = text_prefix + text
            continue
        else:
            if len(text_prefix) < min_len:  # 片段太短， 拼接到上一个或者当前
                if len(split_texts) > 0 and len(split_texts[-1]) < len(text):
                    split_texts[-1] += text_prefix
                    text_prefix = text
                else:
                    split_texts.append(text_prefix+text)
                    text_prefix = ""
            else: # 片段大于最小值， 单独成一个
                split_texts.append(text_prefix)
                text_prefix = text
    if len(text_prefix) > 0:
        split_texts.append(text_prefix)
    return split_texts                
                    
                    

    

class CutSentence(object):
    def __init__(self, min_len = 10, max_len=40):
        self.min_len = min_len
        self.max_len = max_len

        self.punc_map = Punctuation.punc_map
        self.punctuation_map_pattern = '|'.join([re.escape(key) for key in Punctuation.punc_map.keys()])
        self.order_number_pattern = rf"({Punctuation.order_number_prefix})( *\d+)({Punctuation.order_number_suffix})"
        self.split_punc = Punctuation.split_punc
        self.special_punc = Punctuation.special_punc

        self.punc_level1_pattern = "(.*?)([" + re.escape(self.split_punc[0]) + "])"
        self.punc_level2_pattern = "(.*?)([" + re.escape(self.split_punc[1]) + "])"
        self.punc_count_pattern = (r"(?:(?![\u4e00-\u9fff]|[a-zA-Z]+\b|\d+)[^\w\s\u4e00-\u9fff])")
        punc_first = self.split_punc[0] + self.split_punc[1]
        self.first_punc_pattern = "(.*?)([" + re.escape(punc_first) + "])"
        
        self.all_split_punc = "".join(self.split_punc)
        self.serial_number_pattern = "^(\d\d\d\d+)([" + re.escape(self.all_split_punc) + "])"
        
        self.first_min_len = 10 #第一个切片最小长度
        

    
    def special_punc_count(self, text):
        matches = re.findall(self.punc_count_pattern, text)
        
        new_punc = set()
        for match in matches:
            if match not in self.special_punc:
                new_punc.add(match)
        if len(new_punc) > 0:
            logger.info(f'############ add special punc: {new_punc}')
            
    
    def punc_map_process(self, text: str):
        def map_punctuation(match):
            original_punctuation = match.group(0)
            return self.punc_map.get(original_punctuation, original_punctuation)

        mapped_text = re.sub(self.punctuation_map_pattern, map_punctuation, text)
        logger.debug(f'punc_map_process: {text} -> {mapped_text}')
        return mapped_text
            

    def check_split(self, text: str, split: list, msg: str):
        ori_len = len(text)
        split_len = sum([len(s) for s in split])
        if split_len != ori_len:
            logger.info(f'<----------------------- split error  {msg} ------------------')
            logger.error(f'split_len: {split_len}, ori_len: {ori_len}')
            logger.error(f'text: \n{text}')
            logger.error(f'split: \n{split}')
            logger.info('----------------------- split error ------------------>')
            

        
    def split_order_number(self, texts:list):
        split_texts = []
        for text in texts:
            split_text = []
            seg_prefix = ""
            matches = list(re.finditer(self.order_number_pattern, text))
            last_end = 0
            for match in matches:
                    start, end = match.span()
                    # 在数字处切分
                    prefix, number, suffix = match.groups()
                    text_seg = seg_prefix +  text[last_end:start] + prefix 
                    seg_prefix = number + suffix
                    if len(text_seg) > 0:
                        split_text.append(text_seg)
                    # 更新上一个匹配结束的位置
                    last_end = end
                    
            text_seg = seg_prefix + text[last_end:]
            if len(text_seg) > 0:
                split_text.append(text_seg)
            self.check_split(text, split_text, "split_order_number")
            split_texts.extend(split_text)
        logger.debug(f'split_order_number: {len(split_texts)} seg, each-len {[len(s) for s in split_texts]}')
        return split_texts
    
    
    def split_by_punc(self, text, punc_str):
        punc_pattern = "(.*?)([" + re.escape(punc_str) + "])"
        split_text = []
        matches = list(re.finditer(punc_pattern, text))
        last_end = 0
        for match in matches:
                start, end = match.span()
                prefix,  punc = match.groups()
                text_seg = prefix + punc 
                if len(text_seg) > 0:
                    split_text.append(text_seg)
                last_end = end
        
        text_seg = text[last_end:]
        if len(text_seg) > 0:
            split_text.append(text_seg)
        
        self.check_split(text, split_text, "split_punc_level1")
        return split_text
    
    def merge_by_len(self, texts:list):
        text_prefix = ""
        split_texts = []
        for text in texts:
            if len(text_prefix + text) > self.max_len :
                split_texts.append(text_prefix)
                text_prefix = text
            else:
                text_prefix = text_prefix + text
        if len(text_prefix) > 0:
            split_texts.append(text_prefix)
        split_texts = [txt for txt in split_texts if len(txt) > 0]
        return split_texts
    
    
    
    def split_punc_level1(self, texts:list):
        split_texts = []
        for text in texts:
            if len(text) < self.max_len:
                split_texts.append(text)
                continue
            split_text = self.split_by_punc(text, self.split_punc[0])
            self.check_split(text, split_text, "split_punc_level1")
            split_texts.extend(split_text)
        split_lens = [len(s) for s in split_texts]
        logger.debug(f'split_punc_level1: {len(split_texts)} seg, total {sum(split_lens)} each-len {split_lens}')
        logger.debug(f'split_punc_level1: {split_texts}')
        return split_texts
    
    def split_punc_level2(self, texts:list):
        split_texts = []
        for i, text in enumerate(texts):
            if len(text) < self.max_len :
                split_texts.append(text)
                continue
            
            split_text = self.split_by_punc(text, self.split_punc[1])
            split_text = self.merge_by_len(split_text)
            self.check_split(text, split_text, "split_punc_level2")
            
            split_texts.extend(split_text)
        split_lens = [len(s) for s in split_texts]
        logger.debug(f'split_punc_level2: {len(split_texts)} seg, total {sum(split_lens)} each-len {split_lens}')
        logger.debug(f'split_punc_level2: {split_texts}')
        return split_texts
    
    # 按第一个标点切分，分成2分
    def split_first_text(self, text:str):
        matches = list(re.finditer(self.first_punc_pattern, text))
        split_text = []
        last_end = 0
        for match in matches:
                start, end = match.span()
                prefix,  punc = match.groups()
                split_text.append(prefix + punc)
                last_end = end
                break
        if last_end != len(text):
            split_text.append(text[last_end:])
        
        return split_text
    
    
    def process_serial_number(self, text):
        texts = self.split_by_punc(text, self.all_split_punc)
        out_text = ""
        for txt in texts:
            matches = list(re.finditer(self.serial_number_pattern, txt.strip()))
            if matches:
                number, punc = matches[0].groups()
                chinese_number = "".join([serial_number_map[x] for x in number])
                out_text = out_text + chinese_number + punc
            else:
                out_text = out_text + txt
        return out_text
    
    # 特殊处理
    def special_process(self, text: str):
        in_text = text
        logger.debug(f'special_process in: {text}')
        text = text.strip()
        
        # 1. 去除转义符
        text = re.sub(r"\\", "", text)
        text = re.sub(r"\\/", "/", text)
        text = re.sub(r"\\n", " ", text)
        logger.debug(f'special_process step1: {text}')
        
        
        # 2. 将句首 1） --> 1、
        pattern = rf"^( *\d+)([）\)])(.*)"
        matches = list(re.finditer(pattern, text))
        if matches:
            number, punc, suffix = matches[0].groups()
            text = number + "、" + suffix
        logger.debug(f'special_process step2: {text}')
        
        
        
        
        # 3. 句首数字 中间插入空格
        pattern = rf"^(\d+)(.*)"
        matches = list(re.finditer(pattern, text))
        if matches:
            number, suffix = matches[0].groups()
            text = ' ' + number + suffix
        logger.debug(f'special_process step3: {text}')
        
        # 4. 序列号转换
        text = self.process_serial_number(text)
        
        # 5. 删除括号
        
        
        
        logger.debug(f'special_process in[{in_text}] out[{text}]')
        return text
    
    def process(self, text):
        logger.info('-----------------------------------------start--------------------')
        logger.info(f'get ori text: {text}')
        self.special_punc_count(text)
        
        
        text = self.punc_map_process(text)
        texts = [txt for txt in text.split('\n') if txt.strip() != '']
        logger.debug(f'punc-map: {texts}')
        
        # 先切分，在特殊处理，在合并
        texts = self.split_by_punc(text, "".join(self.split_punc))
        texts = [self.special_process(txt) for txt in texts]
        texts = ["".join(texts)]
        
        
        texts = self.split_order_number(texts)
        texts = self.split_punc_level1(texts)
        texts = self.split_punc_level2(texts)
        
        # 第一个句子，按标点切分
        text0 = texts[0]
        text0_split = self.split_by_punc(text0, "".join(self.split_punc))
        first_seg = text0_split[0]
        other_seg = ""
        idx = 1
        for i in range(1, len(text0_split)):
            if len(first_seg) > self.first_min_len:
                break
            first_seg += text0_split[i]
            idx = i+1
        if idx < len(text0_split):
            other_seg = ''.join(text0_split[idx:])
            texts = [first_seg] + [other_seg] + texts[1:]
        else:
            texts = [first_seg] + texts[1:]
        logger.debug(f'idx-{idx} first-{first_seg}, other-seg {other_seg}')
        
        # texts = [texts[0]] + self.merge_by_len(texts[1:])
        
        # texts = [self.special_process(txt) for txt in texts]
        texts = [txt for txt in texts if len(txt.strip()) > 0]
        logger.info(f'cut total {len(texts)} segment')
        for i, text in enumerate(texts):
            logger.info(f'[seg-{i}]: {text}')
        seg_lens = [len(x) for x in texts]
        logger.info(f'seg-lens: {seg_lens}')
        logger.info(f'----------------------------------------end-----------------\n\n\n')
        return texts
    
    

# 文本正则的前处理
class TNPreprocessor(object):
    def __init__(self):
        self.name = 'tn_preprocess'
    
    def __call__(self, text: str)-> str:
        text = self.insert_space_in_hanzi_number(text)
        
        return text
        
    def insert_space_in_hanzi_number(self, text):
        text = re.sub(r'([\u4e00-\u9fff])(\d+)', r'\1 \2', text)     # 中文和数字之间插入空格
        text = re.sub(r'([。！？；，、,：])(\d+)', r'\1 \2', text)       #  标点和数字之间插入空格
        text = re.sub(r'([\u4e00-\u9fff])([-—-])([\u4e00-\u9fff])', r'\1至\3', text)        # 减号处理： 如果两边都是中文， 转换为 至
        text = re.sub(r'(\d+)([-—–])([a-zA-Z]+)', r'\1杠\3', text)      #减号处理： 数字英文之间的 减号 读为 杠
        text = re.sub(r'([a-zA-Z]+)([-—–])(\d+)', r'\1杠\3', text)      #减号处理： 英文数字之间的 减号 读为 杠
        
        return text
    

class TNPostProcessor(object):
    def __init__(self, min_len=10, max_len=40):
        self.name = 'tn_TNPostProcessor'
        self.min_len = min_len
        self.max_len = max_len
        self.punc = "".join(Punctuation.split_punc)
        self.bracket_punc = Punctuation.bracket_punc
        self.rm_punc = Punctuation.post_remove_punc
        
    
    def punc_process(self, text):
        text = text.replace(':', ',')
        

        # 删除 ()
        text = text.replace('（', '(')
        text = text.replace('）', ')')
        # pattern = r'\(\s*(?:km|h|%)\s*\)'
        pattern = r'\(\s*(?:' + '|'.join(self.bracket_punc) + r')\s*\)'
        text = re.sub(pattern, '', text)
        text = text.replace('()', '')
        
        # 符号删除
        pattern = '[' + re.escape(''.join(self.rm_punc)) + ']'
        text = re.sub(pattern, '', text)
        
        text = text.replace('、', ',')
        
        return text


    def process(self, texts: list):
        logger.info(f'--texts: {texts}')
        split_texts = [texts[0]] + merge_by_len_range(texts[1:], self.min_len, self.max_len)
        logger.info(f'pre-seg: {split_texts}')
        split_texts = [self.punc_process(txt) for txt in split_texts]
        
        split_texts = [txt for txt in split_texts if len(txt) > 0]
        
        num_split = len(split_texts)
        logger.info(f'post-seg {num_split} segment')
        for i, txt in enumerate(split_texts):
            logger.info(f'[seg-{i} len {len(txt)}]: {txt}')
        logger.info(f'----------------------------------------post-seg-end-----------------\n')
        
        return split_texts
    
    

if __name__ == '__main__':
    
    logger.propagate = True
    
    cut_sentence = CutSentence()

    text = '2024-07-30东莞中心综合诊断如下：1、综合评价：1）中心综合评价得分是88.17分，全国排名15名，管控较好，请继续保持。2、质量方面：1）中心质量得分是87.25分，目标是85分，全国排名14名，管控较好，请继续保持。2）进港超时库存率是6.06%，目标是5.2%，全国排名42名，管控较差，请继续努力。3）重复进线率是4.72%，目标是4%，全国排名57名，管控较差，请继续努力。4）始发破损率是十万分之30.24，目标是十万分之90，全国是十万分之83.26，全国4名，管控较好，请继续保持。东莞,目的 华北,破损量 51, 东莞,目的 佛山,破损量 45, 东莞,目的 揭阳,破损量 41, '
    # text = sys.argv[1]
    
                

            
    
    split_text = cut_sentence.process(text)
    split_lens = [len(s) for s in split_text]
    print(f'final ori-len {len(text)}, split-len {sum(split_lens)}, seg-len {split_lens}')
    # break