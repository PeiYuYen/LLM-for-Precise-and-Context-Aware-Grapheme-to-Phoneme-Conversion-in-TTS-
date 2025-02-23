from ollama import chat
from ollama import ChatResponse
import ollama
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import os
import random
from collections import Counter
from chinese_dict import create_dict_prompt
import re


#sampled_data_100.json
#test_100data.json

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
        
def get_sample_data():
    with open("test_100data.json", "r", encoding="utf-8") as file:
        sample_data = json.load(file)
        sentences = [item["content"] for item in sample_data]
        groundtruths = [item["groundtruth"] for item in sample_data]
        return sentences, groundtruths

def get_cppdata(n=1):
    with open("./concate_cpp/test.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    sample_data = random.sample(data, n)

    with open("test_data.json", "w", encoding="utf-8") as out_file:
        json.dump(sample_data, out_file, ensure_ascii=False, indent=4)
    
        
def calculate_accuracy(predictions, references):
    correct = sum([1 for pred, ref in zip(predictions, references) if pred == ref])
    return correct / len(references) * 100

def calculate_bleu4(predictions, references):
    smooth_func = SmoothingFunction().method1  # 避免分數因未匹配導致為 0
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        ref_tokens = [ref]  
        pred_tokens = pred 
        score = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func)
        bleu_scores.append(score)

    return sum(bleu_scores) / len(bleu_scores) * 100

def save_result(sentences, predicts,  groundtruths):
    os.makedirs("result", exist_ok=True)
    data = []

    for c, p, r in zip(sentences, predicts,  groundtruths):
        data.append({"sentence": c, "predict": p, "groundtruth": r})

    with open(f"./result/result.json", "w") as f:
        json.dump(data, f, indent=5, ensure_ascii=False)
        
def save_wrong_result(contents, predictions, references):
    os.makedirs("result", exist_ok=True)
    data = []

    for c, p, r in zip(contents, predictions, references):
        if p != r:
            data.append({"content": c, "predict": p, "reference": r})

    with open(f"./result/wrong_result.json", "w") as f:
        json.dump(data, f, indent=5, ensure_ascii=False)

def model(question):
    response: ChatResponse = chat(
        model = 'gemma2:27b', 
        options={
            'top_k' : 1.0,
            'temperature' : 0.0,
            'seed': 42,
        },
        messages = [
                    {
                    'role': 'system',
                    'content': """
                    完成多音字判别任务，你必須依照user提供的内容，输出句子中**内该多音字的读音。
                    ###范例###
                    范例一：
                    多音字: 佣，有多个读音:
                    yong1，漢語拼音出現機率為：0.943
                    [动]
                    [名]
                    [形]
                    yong4，漢語拼音出現機率為：0.057
                    [  ]

                    句子如下：
                    他背叛自己雇**佣**的手下。
                    输出：
                    yong1
                    范例二：
                    多音字: 厦，有多个读音:
                    sha4，漢語拼音出現機率為：0.5312
                    [名]
                    [形]
                    xia4，漢語拼音出現機率為：0.4688
                    [名]
                    [  ]

                    句子如下：
                    1993年12月，**厦**门火车站实现电气化。
                    输出：
                    xia4           
                    范例三：
                    多音字: 疟，有多个读音:
                    nu:e4，漢語拼音出現機率為：1.0
                    [名]
                    [动]

                    句子如下：
                    1934年，玛莉安不幸因**疟**疾去世。
                    输出：
                    nu:e4
                    ###注意###
                    你只能输出一个读音，不要输出其他文字或解释。
                    你只能输出user有提供的读音選項而已。
                    你只能输出user所提供的拼音格式，不能簡化或修改。
                    """
                    },
                    {
                        'role': 'user', 
                        'content': question
                    },
            ],
        
        ) 
    
    predict = response.message.content
    
    if " " in predict:
        predict = predict.replace(" ", "") 
    if "\n" in predict:
        predict = predict.replace("\n", "") 
        
    return predict

# def model1(question):
#     response: ChatResponse = chat(
#     model = 'gemma2:27b', 
#     options= {
#         'top_k' : 1.0,
#         'temperature' : 0.0,
#     },
#     messages = [
#                 {
#                     'role': 'system',
#                     'content': """
#                      請從該單詞的漢語拼音資訊，一步一步思考選出**內的漢語拼音。漢語拼音格式應該是數字在拼音之後，例如：bei1，不要在韵母上标记音调。
#                     """
#                 },
#                 {
#                     'role': 'user', 
#                     'content': question
#                 },
#         ]
#     )

#     return response.message.content

# def model2(question):
#     response: ChatResponse = chat(
#     model = 'gemma2:27b', 
#     options= {
#         'top_k' : 1.0,
#         'temperature' : 0.0,
#     },
#     messages = [
#                 {
#                     'role': 'system',
#                     'content': """
#                      ###任务###
#                     根据以上结论，总结出最后应该要输出的汉语拼音。
#                     ###范例###
#                     bei1
#                     ###提示###
#                     你只能输出汉语拼音，不要输出其他文字或解释。
#                     音调应该在单词后面用数字表示。不要在韵母上标记音调。
#                     """
#                 },
#                 {
#                     'role': 'user', 
#                     'content': question
#                 },
#         ]
#     )
    
    predict = response.message.content
    if " " in predict:
        predict = predict.replace(" ", "") 
    if "\n" in predict:
        predict = predict.replace("\n", "") 
    
    return predict


if __name__ == "__main__":
    # dataset_prepocessing()
    num = 1000#10254
    # get_cppdata(num) # 生成出 n 筆後就可以先註解掉了
    sentences, groundtruths = get_sample_data()
    
    predicts = []
    for sentence, groundtruth in tqdm(zip(sentences, groundtruths), total=num, desc="Processing", unit="item"):
        polyphone = sentence.split("**")[1]
        dict_prompt = create_dict_prompt(polyphone, sentence)
        
        # run 5 次
        # predictions = [model(dict_prompt) for _ in range(5)]
        # print(predictions)
        # most_common_predict = Counter(predictions).most_common(1)[0][0]
        
        # run 1 次
        most_common_predict = model(dict_prompt)    
        
        ## 兩步驟處理
        # model_reply = model1(dict_prompt)
        # most_common_predict = model2(model_reply)        
            
        predicts.append(most_common_predict)
        print(sentence)
        print(f'predict: {most_common_predict}, groundtruth: {groundtruth}\n')
        
    
    accuracy = calculate_accuracy(predicts,  groundtruths)
    print(f"Accuracy: {accuracy:.2f}%")
    save_result(sentences, predicts, groundtruths)
    save_wrong_result(sentences, predicts,  groundtruths)
    
    # bleu_score = calculate_bleu4(predicts,  groundtruths)
    # print(f"BLEU-4 Score: {bleu_score:.2f}%")
        
        
        
        