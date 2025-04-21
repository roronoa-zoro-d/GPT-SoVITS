import logging
import os
import re
import sys
from glob import glob

chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")

import shutil
from tqdm import tqdm
from conf import WETEXT_ROOT_DIR

sys.path.append(WETEXT_ROOT_DIR)

from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer

from cut_split import CutSentence, TNPreprocessor, TNPostProcessor
from yto_utils import init_file_logger

logger = init_file_logger("tts_front", logging.DEBUG, propagate=False)


class TTS_Front(object):
    def __init__(self):
        self.tn_fst_dir = f"{WETEXT_ROOT_DIR}/tn"
        self.zh_normalizer = ZhNormalizer(
            cache_dir=self.tn_fst_dir,
            overwrite_cache=False,
            remove_interjections=True,
            remove_erhua=True,
            traditional_to_simple=True,
            remove_puncts=False,
            full_to_half=False,
            tag_oov=False,
        )
        self.en_normalizer = EnNormalizer(
            cache_dir=self.tn_fst_dir, overwrite_cache=False
        )

        self.preprocessor = TNPreprocessor()
        self.cut_sentence = CutSentence(min_len=10, max_len=40)
        self.post_processor = TNPostProcessor(min_len=10, max_len=40)

    def process_origin(self, input_text, split=True):
        text = input_text.strip()
        text = self.preprocessor(text)  # 正则处理
        text = self.zh_normalizer.normalize(text)  # 数字转换等

        text = text.replace("\n", "")
        text = replace_blank(text)  # 删除空格
        text = replace_corner_mark(text)  # 符号转换
        text = text.replace(".", "、").replace(" - ", "，")
        text = remove_bracket(text)  # 删除括号

        norm_texts = split_paragraph(
            text, token_max_n=30, token_min_n=5, merge_len=1, comma_split=True
        )

        logger.debug(f"--------------------------")
        logger.debug(f"{input_text}")
        for norm_text in norm_texts:
            logger.debug(f"{norm_text}")
        logger.debug(f"--------------------------")

        if split is False:
            return text
        return norm_texts

    def process(self, input_text, split=True):
        text = input_text.strip()

        norm_text = self.preprocessor(text)
        texts = self.cut_sentence.process(norm_text)
        norm_texts = []
        for i, text in enumerate(texts):
            # tag_str = self.zh_normalizer.tag(text)
            norm_str = self.zh_normalizer.normalize(text)
            # norm_str = norm_str.strip().replace(' ', '')
            # logger.debug(f'--------------------------')
            # logger.debug(f'idx-{i}: {text}')
            # logger.debug(f'{tag_str}')
            # logger.debug(f'idx-{i}: {norm_str}')
            # norm_texts.append(self.post_text(norm_str))
            norm_texts.append(norm_str)

        # post_norm_texts = self.post_processor.process(norm_texts)
        merge_text = "".join(norm_texts)
        text = self.post_text(merge_text)

        norm_texts = split_paragraph(
            text, token_max_n=30, token_min_n=5, merge_len=1, comma_split=True
        )

        # logger.debug(f"--------------------------")
        # logger.debug(f"{input_text}")
        # for norm_text in norm_texts:
        #     logger.debug(f"{norm_text}")
        # logger.debug(f"--------------------------")

        if split is False:
            return text
        return norm_texts

    def post_text(self, text):
        text = text.replace("\n", "")
        text = replace_blank(text)  # 删除空格
        # text = replace_corner_mark(text) # 面积单位转换
        text = text.replace(".", "、").replace(" - ", "，").replace('──', '、')
        text = remove_bracket(text)  # 删除括号
        text = re.sub(r'\s+', '', text)
        return text


# 判断文本中是否包含中文字符
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# 替换特殊符号，如上标2和3
def replace_corner_mark(text):
    text = text.replace("²", "平方")
    text = text.replace("³", "立方")
    return text


# 移除无意义的符号，如括号和尖括号
def remove_bracket(text):
    text = text.replace("（", "").replace("）", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("【", "").replace("】", "")
    text = text.replace("`", "").replace("`", "")
    text = text.replace("——", " ")
    return text


# 将阿拉伯数字转换为单词拼写
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


# 按段落分割文本# 按段落分割文本
def split_paragraph(
        text: str, token_max_n=80, token_min_n=60, merge_len=20, comma_split=True
):
    # 定义一个内部函数，判断文本是否应该被合并
    def should_merge(_text: str):
        return len(_text) < merge_len

    # 初始化标点符号列表，包括中英文标点
    pounc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";", "，", ",", ":"]

    # 初始化起始索引和分割后的文本列表
    st = 0
    utts = []
    # 遍历文本每个字符，根据标点符号进行分割
    for i, c in enumerate(text):
        if c in pounc:
            # 遇到标点符号且分割长度大于0时，将文本添加到列表中
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            # 如果下一个字符是引号，则合并，并更新起始索引
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1
        elif i == len(text) - 1:
            print(i, c, st)
            utts.append(text[st:i] + c + "、")
    # 如果分割后的文本列表为空，添加原文本到列表中
    if len(utts) == 0:
        utts.append(text + "。")

    # 初始化最终分割后的文本列表和当前段落文本
    final_utts = []
    cur_utt = ""
    # 遍历分割后的文本，根据长度进行进一步分割
    for utt in utts:
        if (
                len(utt) == 2
                and utt[0] in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
                and utt[1] == "、"
        ):
            final_utts.append(cur_utt)
            cur_utt = ""
            cur_utt = cur_utt + utt
            continue

        # 如果当前段落加上新的句子长度超过最大长度，且当前段落长度大于最小长度，则将当前段落添加到最终列表中
        if len(cur_utt + utt) > token_max_n and len(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    # 处理最后一个段落
    if len(cur_utt) > 0:
        # 如果最后一个段落长度小于合并长度且最终列表不为空，则将其合并到最后一个段落
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


# 替换不当的空格，仅在ASCII字符间保留空格
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if (text[i + 1].isascii() and text[i + 1] != " ") and (
                    text[i - 1].isascii() and text[i - 1] != " "
            ):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def convert_text():
    name = '维维'
    data_path = f'/data/nas/dataset/tts/huoshan3_0912/{name}-0912'
    output_path = f'/data/nas/aim2/tts/hs'

    wav_files = sorted(glob(f'{data_path}/*.wav'))
    print(f'len:{len(wav_files)}')
    for i in tqdm(range(len(wav_files))):
        wav_file = wav_files[i]
        file_name = os.path.splitext(os.path.split(wav_file)[1])[0]
        txt_file = f'{data_path}/{file_name}.txt'

        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                line = f.readline().strip().replace('\n', '')
                line = tts_front.process(line, split=False)
                shutil.copyfile(wav_file, f'{output_path}/{name}_{file_name}.wav')
                with open(f'{output_path}/{name}_{file_name}.txt', 'w') as f:
                    f.write(line)


if __name__ == "__main__":
    logger.propagate = True

    tts_front = TTS_Front()
    text = "未来，圆通只人将做坚信之人，行分享之事，让圆通更智慧，让员工更幸福，让人生更精彩。"
    text = "网点的盈利状况，是由收入提升和成本控制两方面决定的，圆通网点要想赚钱，主要是做好“三升四降”。1、提升服务质量，网点可以通过缩短揽派时长、提高出港交件及时率和签收及时率，努力减少遗失破损和虚假签收等举措来提升服务质量。2、提升客户体验，网点要重点推广好直通总部的快速理赔功能，提升理赔速度，同时要努力降低重复进线率和缩短专属群回复时长，服务质量和客户体验是决定快递价格差异的核心要素。"
    text = "2024-07-31 顾霞的客户综合分析如下：1、发货方面：1)业务量为10票，上期业务量为9.00票，管控较好。2、服务质量：1）发出维度ccr是万分之1000，全国均值为万分之79.63，管控较差。"
    text = "2024-08-01郑州中心进港诊断：1、超时库存：进港离场超时库存27243票，当前未解决11695票。379001,网点名称 河南省洛阳市,当日超时未解决票数 461, 371069,网点名称 河南省郑州市龙湖,当日超时未解决票数347， 371050,网点名称 河南省郑州市中牟县,当日超时未解决票数 308, 2、进港卸车：1）早班待卸车6辆，中班待卸车4辆；实时待卸车5辆，平均等待时长0.59h，等待时长超1小时1辆，待卸87419票，待卸7910件，待卸2846包。AQ951905006954,始发 -,等待时长 1.76, 2）进港卸车平均卸车效率1476.00。3）未来0-1小时到达车3辆，1231包，4463件单件，31061票。1-2小时2辆，916包，3442件单件，27307票。2小时以上49辆，26147包，102273件单件，1009847票。3、进港拆包：1）早班待拆包449个，中班待拆包177个，实时待拆包198个,平均滞留时长0.17h；滞留超2小时以上1包。NW21210156292,建包单位 河北省保定市高碑店市白沟镇,拆包单位 郑州转运中心,滞留时长(h) 3.37, 2）进港分拣操作票量804701，平均操作效率1757.00件\/h，最新操作效率1755.00件\/h。1-DWS-18,最新操作效率 837, 1-DWS-3,最新操作效率 1292, 1-DWS-15,最新操作效率 1304,3）进港小循环拥堵告警0次，累计告警1次，请注意控制小循环流量。4）进港未建包量712，当日累计未建包量8395，累计占比36.31%，请注意管控，严禁单件进入下包线。4、进港上车：1）进港错装55票，进港错装率0.31%。2）拉包不码货告警0次，累计告警4次，请注意网点车位码货。郑州,最新告警次数 1216,累计告警次数 38, 3）进港车等货累计告警92次。河南省三门峡市义马市,累计告警次数 28,最新告警次数 1, 河南省郑州市郑东新区职教园,累计告警次数 16,最新告警次数 1, 河南省郑州市金水东区,累计告警次数16,最新告警次数 1。"
    text = "2024-07-30贵阳中心综合诊断如下：1、综合评价：1）中心综合评价得分70.2分，全国排名57名，管控较差，请继续努力。2、质量方面：1）中心质量得分是53.42分，目标是85分，全国64名，管控较差，请继续努力。2）遗失率是十万分之3.68，目标是十万分之2.9，全国排名62名，管控较差，请继续努力。3）出港超时库存率是4.94%，目标是2.47%，全国排名71名，管控较差，请继续努力。4）进港超时库存率是7.7%，目标是5%，全国排名52名，管控较差，请继续努力。5）CCR是59.92‱，目标是24‱，全国排名72名，管控较差，请继续努力。6）漏扫描率是0.24%，目标是0.2%，全国排名67名，管控较差，请继续努力。7）始发破损是十万分之947.11，目标是十万分之90，全国排名72名，管控较差，请继续努力。贵阳,目的 哈尔滨,破损量 303, 贵阳,目的 沈阳,破损量 251, 贵阳,目的 华北,破损量 239, 8）倒包不彻底量是7票，请及时关注处理；3、成本方面：1）单票运能成本是0.81元，目标是0.809元，全国排名65名，管控较差，请继续努力。贵阳,目的中心 海南省三亚市直营交换站,出港运能成本 1289.84,出港量 222,单票成本 5.81, 贵阳,目的中心 拉萨,出港运能成本 235.42,出港量 57,单票成本 4.13, 贵阳,目的中心 海口,出港运能成本 5208.58,出港量 1262,单票成本 4.127, 2）单次操作成本是0.218元，目标是0.163元，全国排名68名，管控较差，请继续努力。4、效率方面：1）出港建包平均操作效率是1127票\/h，目标是1300票\/h，全国排名59名，管控较差，请继续努力。2）出港收件平均操作效率是1305件\/h，目标是1180件\/h，全国是1249件\/h，全国排名28名，管控较好，请继续保持。3）进港卸车平均操作效率是1361件\/h，目标是1430件\/h，全国排名42名，管控较差，请继续努力。4）进港拆包平均操作效率是1714票\/h，目标是1700票\/h，全国是1710票\/h，全国排名34名，管控较好，请继续保持。5、人员方面：1）小时工占比是0（≤5%），固定工出勤率是83.6%(＜86%)，人均效能是1895.5，目标是1909，全国排名19名，需杜绝使用小时工；2）人员流失率当月累计是6.31%，全国是6.6%，全国排名56名，管控较好，请继续保持。上个月是11.42%，环比趋势向好，请继续保持；中心操作,月初在职 0,月末在职 0,入职 0,离职 0,流失率 0, 中心操作储备,月初在职 8,月末在职 19,入职 5,离职 0,流失率 0, 3）稳定性指数是39，全国是45.2，全国排名57名，管控较差，请继续努力。中心操作储备,在职 19,平均工龄 3.3,稳定性指数 9.2, 6、安全与目视化：1）装车不规范：告警21次，请及时关注；2）月台下站人：告警8次，请及时关注；3）车等货监控：告警377次，请及时关注；4）拉包不码货告警2次，累计告警40次，请注意网点车位码货。"
    text = "1）中国猿人（全名为“中国猿人北京种”，或简称“北京人”）在我国的发现，是对古人类学的一个重大贡献。（2）写研究性文章跟文学创作不同，不能摊开稿纸搞“即兴”。（其实文学创作也要有素养才能有“即兴”。）"
    text = "根据研究对象的不同，环境物理学分为以下五个分支学科：──环境声学；──环境光学；──环境热学；──环境电磁学；──环境空气动力学。"
    text = '二零二四年八月三十日滇西中心质量分析如下：一，中心质量：一中心质量得分是九十一点三八分。'
    # convert_text()
    norm_texts = tts_front.process(text, split=False)
    print(norm_texts)
    exit()
    for text in norm_texts:
        print(text)
