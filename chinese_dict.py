import pandas as pd
from IPython.display import display
import json
import os
import re

def dataset_prepocessing():
    os.makedirs("concate_cpp", exist_ok=True)
    test_sent_file = "./cpp_dataset/test.sent"
    test_lb_file = "./cpp_dataset/test.lb"
    test_output_file = "./concate_cpp/test.json"
    train_sent_file = "./cpp_dataset/train.sent"
    train_lb_file = "./cpp_dataset/train.lb"
    train_output_file = "./concate_cpp/train.json"

    with open(test_sent_file, "r", encoding="utf-8") as f:
        sentences = [line.strip().replace("▁", "**") for line in f]
    with open(test_lb_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f]
        
    assert len(sentences) == len(labels), "兩個檔案的行數不一致"
    data = [{"content": sent, "groundtruth": lb} for sent, lb in zip(sentences, labels)]
    with open(test_output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)    


    with open(train_sent_file, "r", encoding="utf-8") as f:
        sentences = [line.strip().replace("▁", "**") for line in f]
    with open(train_lb_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f]   
        
    assert len(sentences) == len(labels), "兩個檔案的行數不一致"
    data = [{"content": sent, "groundtruth": lb} for sent, lb in zip(sentences, labels)]
    with open(train_output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)  

def extract_character_data(polyphone):
    df = pd.read_csv('dictionary_zdic_test1.csv')
    filtered_df = df[df['character'] == polyphone]
    return filtered_df

def map_part_of_speech(polyphone):
    df = extract_character_data(polyphone)
    mapping = {
    0: '  ',
    1: '代',
    2: '副',
    3: '连',
    4: '助',
    5: '形',
    6: '名',
    7: '叹',
    8: '数',
    9: '介',
    10: '动',
    11: '量'
    }
    
    df['part_of_speech'] = df['part_of_speech'].fillna(0).astype(int)
    df['part_of_speech'] = df['part_of_speech'].map(mapping)
    return df


# def create_dict_prompt(polyphone, sentence):
#     output = f"多音字: {polyphone}，有多个读音:\n"
#     df_character_mapped = map_part_of_speech(polyphone)
#     grouped = df_character_mapped.groupby("pinyin")

#     for pinyin, group in grouped:
#         output += f"{pinyin}"
#         for _, row in group.iterrows():
#             output += f"   [{row['part_of_speech']}]  {row['definition']}\n"
            
#     dict_prompt = output + f"\n句子如下：\n{sentence}"
#     return dict_prompt

# prompt1
# def create_dict_prompt(polyphone, sentence):
#     output = f"多音字: {polyphone}，有多个读音:\n"
#     df_character_mapped = map_part_of_speech(polyphone)
#     grouped = df_character_mapped.groupby("pinyin")
    
#     df = pd.read_csv("pinyin_ratio.csv")
#     filtered_df = df[df['char'] == polyphone]
#     pinyin_list = []

#     for pinyin, group in grouped:
#         output += f"{pinyin}"
#         pinyin_list.append(pinyin)
#         for _, row in group.iterrows():
#             output += f"   [{row['part_of_speech']}]  {row['definition']}\n"
            
#     for _, ratio_row in filtered_df.iterrows():            
#         if ratio_row['pinyin'] in pinyin_list:
#             output += f"\n漢語拼音 {ratio_row['pinyin']} 出現機率為：{ratio_row['ratio']}"
                
#     dict_prompt = output + f"\n句子如下：\n{sentence}"
#     return dict_prompt

def extract_sentences(text):
    """提取 '如:' 後的完整句子，直到 '。' 為止，若無則返回原始內容"""
    pattern = r'如:(.*?。)'
    matches = re.findall(pattern, text)
    return "; ".join(matches) if matches else text  # 若匹配多個，則用分號連接

#prompt2    
def create_dict_prompt(polyphone, sentence):
    output = f"多音字: {polyphone}，有多个读音:\n"
    df_character_mapped = map_part_of_speech(polyphone)
    grouped = df_character_mapped.groupby("pinyin")
    
    df = pd.read_csv("train_pinyin_ratio.csv")
    filtered_df = df[df['char'] == polyphone]
    
    for pinyin, group in grouped:
        # output += f"{pinyin}"
        for _, ratio_row in filtered_df.iterrows():
            if ratio_row['pinyin'] == pinyin:
                output += f"{pinyin}，漢語拼音出現機率為：{ratio_row['ratio']}\n"
                for _, row in group.iterrows():
                    output += f"[{row['part_of_speech']}]\n"
                    # row['definition'] = extract_sentences(row['definition'])
                    # output += f"[{row['part_of_speech']}]   {row['definition']}\n"
            # else:
            #     output += f"{ratio_row['pinyin']}，，漢語拼音出現機率為：{ratio_row['ratio']}"
            
    dict_prompt = output + f"\n句子如下：\n{sentence}"
    return dict_prompt
    


if __name__ == "__main__":
    # sentence = "**嗯**，我的名字是邓月薇，我是11号航班上的3号空服员。" # in cpp only
    # sentence = "他认为哲学的目的不在于埋**头**苦究“有时间性的瞬即消失的假象”，而是去追求现有事物中永恒的东西。" # some pinyin in cpp 
    #sentence = "西南风向的风切变令风暴的风速没能超过每小时75公里，不过风切变和登陆都没有明显打乱气**旋**的组织结构。" # both
    sentence = "天**姥**山得名于“王母”。"
    # sentence = "莘庄耶稣堂位于中国上海市闵行区莘庄镇**莘**浜路551号莘庄公园西侧，为一座基督新教教堂。"
    #sentence = "短片获得了非常**正**面的评价。"
    # sentence = "解放军攻占**凉**城县后，他留下工作。" # in dict only
    polyphone = sentence.split("**")[1]
    dict_prompt = create_dict_prompt(polyphone, sentence)
    print(dict_prompt)




