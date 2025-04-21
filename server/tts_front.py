import sys
import os
import logging
import traceback
import argparse
from conf import WETEXT_ROOT_DIR
sys.path.append(WETEXT_ROOT_DIR)

from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
from itn.main import str2bool


from cut_split import CutSentence, TNPreprocessor, TNPostProcessor
from yto_utils import init_file_logger

logger = init_file_logger('tts_front', logging.DEBUG, propagate=False)



class TTS_Front(object):
    def __init__(self):
        self.tn_fst_dir = f'{WETEXT_ROOT_DIR}/tn'
        self.zh_normalizer = ZhNormalizer(
                                    cache_dir = self.tn_fst_dir,
                                    overwrite_cache = False,
                                    remove_interjections = False,
                                    remove_erhua = False,
                                    traditional_to_simple = True,
                                    remove_puncts = False,
                                    full_to_half = True,
                                    tag_oov = False
                                )
        self.en_normalizer = EnNormalizer(cache_dir=self.tn_fst_dir,
                                    overwrite_cache=False)
        
        self.preprocessor = TNPreprocessor()
        self.cut_sentence = CutSentence(min_len=10, max_len=40)
        self.post_processor = TNPostProcessor(min_len=10, max_len=40)
    def process(self, text):
        ori_text = text
        text = self.preprocessor(text)
        print(f'1 = {text}')
        texts = self.cut_sentence.process(text)
        print(f'2 = {texts}')

        norm_texts = []
        for i, text in enumerate(texts):
            try:
                # tag_str = self.zh_normalizer.tag(text)
                norm_str = self.zh_normalizer.normalize(text)
            except Exception as e:
                logger.error(f'tn error: {e}')
                logger.error(f'ori-error-text: {text}')
                norm_str = text
            norm_str = norm_str.strip().replace(' ', '')
            logger.debug(f'--------------------------')
            logger.debug(f'idx-{i}: {text}')
            #logger.debug(f'{tag_str}')
            logger.debug(f'idx-{i}: {norm_str}')
            norm_texts.append(norm_str)
            
        post_norm_texts = self.post_processor.process(norm_texts)
        
        
        logger.info(f'------------------ TN {len(post_norm_texts)} segment -----------------')
        logger.info(f'---ori-text: {ori_text}')
        for i, txt in enumerate(post_norm_texts):
            logger.debug(f'[seg-{i}]: {txt}')
        
        
        return post_norm_texts
    
    
    
def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="if set, use text ",
    )
    
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="if set, use text file generate utt.txt",
    )
    
    parser.add_argument(
        '--out_file',
        type=str,
        default='',
        help="output file",
    )
    

    return parser


if __name__ == "__main__":
    logger = init_file_logger('tts_front_test', logging.DEBUG, propagate=True)
    # logger.propagate = True
    
    args = get_parser().parse_args()
    out_file = args.out_file
    # text = sys.argv[1]
    # text = '2024-07-30东莞中心综合诊断如下：1、综合评价：1）中心综合评价得分是88.17分，全国排名15名，管控较好，请继续保持。2、质量方面：1）中心质量得分是87.25分，目标是85分，全国排名14名，管控较好，请继续保持。2）进港超时库存率是6.06%，目标是5.2%，全国排名42名，管控较差，请继续努力。3）重复进线率是4.72%，目标是4%，全国排名57名，管控较差，请继续努力。4）始发破损率是十万分之30.24，目标是十万分之90，全国是十万分之83.26，全国4名，管控较好，请继续保持。东莞,目的 华北,破损量 51, 东莞,目的 佛山,破损量 45, 东莞,目的 揭阳,破损量 41, '
    texts = []
    
    if args.text:
        texts.append(args.text)
    else:
        text_file = args.text_file
        with open(text_file, 'r') as f:
            for line in f:
                txt = line.strip()
                if len(txt) > 0:
                    texts.append(txt)
                
                
    tts_front = TTS_Front()

    try:
        res = []
        for txt in texts:
            norm_texts = tts_front.process(txt)
            res.append([txt, norm_texts])
        
        if out_file != "":
            with open(out_file, 'w') as f:
                for ori_txt, segs in res:
                    num_seg = len(segs)
                    f.write(f'------------ tts-front-------\n')
                    f.write(f'---total {num_seg} seg,  \n')
                    f.write(f'---ori: {ori_txt}\n')
                    for i, txt in enumerate(segs):
                        f.write(f'[seg-{i}]: {txt}\n')
                    f.write(f'------------ tts-front-------\n\n')
            
    except Exception as e:
        traceback.print_exc()
