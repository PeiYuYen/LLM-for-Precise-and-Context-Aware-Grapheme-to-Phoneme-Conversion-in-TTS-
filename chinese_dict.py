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
    df = pd.read_csv('dictionary_zdic_v2preview.csv')
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

def example_sentences(data, target_label, polyphone):
    sentences = [
        item["sentence"] for item in data 
        if item["label"] == target_label and polyphone in item["sentence"]
    ]
    return "\n".join(sentences[:5]) if sentences else "無對應句子"

#prompt2    
def create_dict_prompt(polyphone, sentence):
    output = f"多音字: {polyphone}，有多个读音:\n"
    df = pd.read_csv("polyphone_list.csv", names=["Character", "Pinyin"])
    most_common = df[df["Character"] == polyphone]["Pinyin"].values[0]
    # output = f"多音字: {polyphone}，其中最常使用的為 {most_common}，供以參考。\n有多个读音 : \n"

    df_character_mapped = map_part_of_speech(polyphone)
    grouped = df_character_mapped.groupby("pinyin")

    df = pd.read_csv("train_pinyin_ratio.csv")
    filtered_df = df[df['char'] == polyphone]

    with open('train_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for pinyin, group in grouped:
        # output += f"{pinyin}"
        for _, ratio_row in filtered_df.iterrows():
            if ratio_row['pinyin'] == pinyin:
                output += f"{pinyin}，漢語拼音出現機率為：{ratio_row['ratio']}\n"
                # output += f"{pinyin}"
                # if pinyin == most_common:
                #     output += " (這是最常見的读音)\n" 
                # else:
                #     output += "\n"
                for _, row in group.iterrows():
                    output += f"[{row['part_of_speech']}]\n"
                    # output += f"[{row['part_of_speech']}]   {row['definition']}\n"
                    # output += f"[{row['part_of_speech']}] "
                output += f"example: {example_sentences(data, pinyin, polyphone)}\n"
            # else:
            #     output += f"{ratio_row['pinyin']}，，漢語拼音出現機率為：{ratio_row['ratio']}"
            
    dict_prompt = output + f"\n句子如下：\n{sentence}"
    return dict_prompt
    


if __name__ == "__main__":
    # sentence = "**嗯**，我的名字是邓月薇，我是11号航班上的3号空服员。" # in cpp only
    # sentence = "他认为哲学的目的不在于埋**头**苦究“有时间性的瞬即消失的假象”，而是去追求现有事物中永恒的东西。" # some pinyin in cpp 
    # sentence = "西南风向的风切变令风暴的风速没能超过每小时75公里，不过风切变和登陆都没有明显打乱气**旋**的组织结构。" # both
    # sentence = "1934年，玛莉安不幸因**疟**疾去世。"
    #sentence = "第二天，卡洛塔死在家中，而波洛在却她的包里发现了她化妆成简所需要的**假**发等物品。"
    # sentence = "义成碶位于北仑区戚家山街道**蔚**斗社区，长32米，宽5."
    # sentence = "**蠡**测具有属地、数量、日期、祭祀等寓意。"
    # sentence = "她被关在棺材里，被带到地下最后一个拉**撒**路池，以便仪式开始。"
    sentence = "2013年12月28日，3号线工程在盘**蠡**路站举行动工仪式。"
    # sentence = "解放军攻占**凉**城县后，他留下工作。" # in dict only
    polyphone = sentence.split("**")[1]
    dict_prompt = create_dict_prompt(polyphone, sentence)
    print(dict_prompt)




