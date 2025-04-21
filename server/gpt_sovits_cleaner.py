import sys
from conf import GPT_ROOT_DIR
sys.path.append(f'{GPT_ROOT_DIR}/')
print(sys.path)
from text import chinese, japanese, cleaned_text_to_sequence, symbols, english

print(chinese.__file__)
language_module_map = {"zh": chinese, "ja": japanese, "en": english}
special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]

# 正则+g2p
class GPTCleaner(object):
    def __init__(self,):
        self.language_module_map = {"zh": chinese, "ja": japanese, "en": english}
        
    



    def clean_text(self, text, language):
        if(language not in self.language_module_map):
            language="en"
            text=" "
        for special_s, special_l, target_symbol in special:
            if special_s in text and language == special_l:
                return self.clean_special(text, language, special_s, target_symbol)
        language_module = self.language_module_map[language]
        norm_text = language_module.text_normalize(text)
        if language == "zh":
            phones, word2ph = language_module.g2p(norm_text)
            assert len(phones) == sum(word2ph)
            assert len(norm_text) == len(word2ph)
        elif language == "en":
            phones = language_module.g2p(norm_text)
            if len(phones) < 4:
                phones = [','] * (4 - len(phones)) + phones
            word2ph = None
        else:
            phones = language_module.g2p(norm_text)
            word2ph = None

        for ph in phones:
            assert ph in symbols, f'ph {ph} not in symbols'
        return phones, word2ph, norm_text


    def clean_special(self, text, language, special_s, target_symbol):
        """
        特殊静音段sp符号处理
        """
        text = text.replace(special_s, ",")
        language_module = self.language_module_map[language]
        norm_text = language_module.text_normalize(text)
        phones = language_module.g2p(norm_text)
        new_ph = []
        for ph in phones[0]:
            assert ph in symbols
            if ph == ",":
                new_ph.append(target_symbol)
            else:
                new_ph.append(ph)
        return new_ph, phones[1], norm_text


    def text_to_sequence(self, text, language):
        phones = self.clean_text(text)
        return cleaned_text_to_sequence(phones)






if __name__ == "__main__":
    text = sys.argv[1]
    gpt_cleaner = GPTCleaner()
    # print(clean_text("你好%啊,啊^啊额、还是到付红四方。", "zh"))
    print(gpt_cleaner.clean_text(text, "zh"))
    
