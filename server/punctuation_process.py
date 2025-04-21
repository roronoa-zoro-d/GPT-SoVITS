import sys
import os
import re



# 合并连续的标点符号
def remove_consecutive_space(text):
    while "  " in text:
        text = text.replace("  ", " ")
    return text



def punctuation_process(text, language):
    text = remove_consecutive_space(text)
    return text