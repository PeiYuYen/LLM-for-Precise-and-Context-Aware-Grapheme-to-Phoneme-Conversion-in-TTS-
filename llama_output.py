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


#sampled_data_100_v1
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
    with open("sampled_data_1000.json", "r", encoding="utf-8") as file:
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
        model = 'deepseek-r1:14b', 
        options={
            'top_k' : 1.0,
            'temperature' : 0.0,
            'seed': 42,
            "num_ctx": 5028
        },
        messages = [
                    {
                    'role': 'system',
                    'content': """
                    完成多音字判别任务，你必須依照用户提供的内容，输出句子中**内该多音字的读音。
                    ###注意###
                    你只能输出用户有提供的读音選項而已。
                    你只能输出用户所提供的拼音格式，不能簡化或修改。
                    不要在韻母標記音調，例如："zhèng4"。
                    你必须將最後的答案輸出在<answer>標籤中。
                    ###范例1###
                    多音字: 佣，有多个读音:
                    yong1，漢語拼音出現機率為：0.943
                    [动]
                    [名]
                    [形]
                    example: 埃尔哈特的经济状况有所好转，他们买了一幢新房子，雇了两个**佣**人，但是埃德温酗酒的恶习很快显露出来。
                    其不一定会带有过于明显的倾向，也未必是职业的，受人雇**佣**。
                    《海牙公约》并没有明确提到雇佣军,但是《海牙第五公约》有关中立的规定则蕴涵了雇**佣**军的活动。
                    雇**佣**保险于1995年开始实行。
                    但刘易斯又去找回薇薇安，并用3000美元雇**佣**她在接下来的一周当自己的临时女友。
                    yong4，漢語拼音出現機率為：0.057
                    [  ]
                    example: 过去，媒体占到85%，剩下的15%付给广告代理，作为固定的**佣**金。
                    出售股权时，担保人会保留收益的一部分作为**佣**金。
                    截至2018年底，就股票经纪**佣**金而言，该公司在韩国经纪市场的份额约为10.
                    券商和银行的加入，通常是为了拉拢新入市的交易者在自己或相关的机构开户，赚取**佣**金、交易费等。
                    网站通常鼓励发展下线会员，并能从下线会员抽取**佣**金。

                    句子如下：
                    他背叛自己雇**佣**的手下。

                    输出：
                    <think>
                    好，我现在需要解决这个多音字判别任务。用户给了一个句子：“他背叛自己雇佣的手下。”，其中“佣”是需要判断的多音字，并且已经提供了两个拼音选项：yong1和yong4。
                    首先，我要理解每个拼音的具体含义和用法。根据提供的例子，“佣”在yong1的时候通常作动词，比如“雇佣”，表示雇用的意思；而yong4则多用于名词或形补，比如“佣金”，表示酬劳。根据句子结构，“雇佣”的组合中，前面已经有“雇”这个动词，后面跟的应该是作为名词或者动词补充的部分。
                    接下来分析句子中的具体使用。“他背叛自己雇佣的手下。”这里的“佣”跟在“雇”之后，可能构成一个双音节动词“雇佣”，即yong1。这种情况下，“雇佣”是一个整体，表示雇用的意思，符合句意中背叛手下，这些手下是被他雇佣的。
                    另外，用户还提供了几个例句来说明每个拼音的用法和常见程度。根据这些例句，如果考虑yong4的情况，通常会和名词搭配，比如“佣金”。但在这个句子中，并没有这样的搭配，而是直接跟在“雇”之后，作为动词的一部分。因此，这里应该是使用yong1的读音。
                    最后，按照要求，我只需要输出提供的选项中的正确读音，并且不要加音调。所以答案就是yong1。
                    </think>
                    <answer>yong1</answer>
                    ###范例2###
                    多音字: 莘，有多个读音:
                    shen1，漢語拼音出現機率為：0.604
                    [形]
                    [名]
                    example: 大学期间，郑孝燮成绩优异，先后获得“中国营造学社桂**莘**奖学金”和“基泰工程司”奖学金。
                    1961年，恢复馆陶、阳谷、东阿、**莘**县4县；德州市及夏津、平原、齐河3县划归德州专区。
                    1946年1月发生张**莘**夫事件。
                    1128年，攻打宋朝，取阳谷、**莘**县、海州。
                    南临黄河，与山东省鄄城县相望，北依金堤，和山东**莘**县接壤，东毗台前县，西接濮阳县。
                    xin1，漢語拼音出現機率為：0.396
                    [名]
                    example: 1966年5月经国务院批准，**莘**庄镇为县属城镇，脱离公社，成立莘庄镇人民委员会。
                    项目西侧为商业区和交通枢纽，真正坐落于原**莘**庄站站址上方。
                    上海交通大学七宝校区是上海交通大学的一个校区，位于上海市闵行区七**莘**路2678号。
                    自上海南站起票价至龙华1角，至**莘**庄2角，至新桥3角，至松江4角。
                    莘庄耶稣堂位于中国上海市闵行区莘庄镇莘浜路551号**莘**庄公园西侧，为一座基督新教教堂。

                    句子如下：
                    虹**莘**路站位于上海闵行区顾戴路虹莘路，为上海轨道交通12号线的地下车站。

                    输出：
                    <think>
                    好，我现在需要解决这个多音字判别任务。用户给了一个句子：“虹**莘**路站位于上海闵行区顾戴路虹莘路，为上海轨道交通12号线的地下车站。”，其中“莘”是需要判断的多音字，并且已经提供了两个拼音选项：shen1和xin1。
                    首先，我要理解每个拼音的具体含义和用法。根据提供的例子，“莘”在shen1时用于地名，比如“莘县”，而在xin1时也用于地名，如“莘庄”。这意味着“莘”在不同的语境中有两种不同的发音。
                    接下来，看给定的句子：“虹**莘**路站位于上海闵行区顾戴路虹莘路，为上海轨道交通12号线的地下车站。”这里的“莘”显然是作为地名的一部分出现的。因为“莘”在这里指代的是上海闵行的一个区域或地点，所以需要判断是shen1还是xin1。
                    另外，用户还提供了几个例句来说明每个拼音的用法和常见程度。根据这些例句，当“莘”用于表示像“莘庄”这样的地方时，它用的是xin1。例如：“上海交通大学七宝校区……七**莘**路2678号。”这里的“莘”显然是读作xin1。“莘庄”是一个常见的地名，发音为xin1。
                    此外，在例句中，“莘县”用shen1，而“莘庄”则用xin1。因此，在句子“虹莘路站……”中，“莘”显然是指“莘庄”，所以正确的拼音应该是xin1。
                    最后，按照要求，我只需要输出提供的选项中的正确读音，并且不要加音调。所以答案就是xin1。
                    </think>
                    <answer>xin1</answer>
                    ###范例3###
                    多音字: 蠡，有多个读音:
                    li2，漢語拼音出現機率為：0.0559
                    [名]
                    example: 北匈奴单于派左谷**蠡**王率领两万骑兵进攻车师。
                    左谷**蠡**王 (狐鹿姑单于之子)
                    各有分地，左谷**蠡**王驻匈奴东部，右谷蠡王驻西部。
                    左贤王、右谷**蠡**王以不得立怨恨，率其众想要南归汉朝。
                    狐鹿姑单于死时立下遗言，因为其子年少，所以由其弟右谷**蠡**王继位为单于。
                    li3，漢語拼音出現機率為：0.9441
                    [动]
                    [  ]
                    example: 沈岭（），直隶**蠡**县人（今河北蠡县）。
                    蒋尔恂，是清初河北**蠡**县生员，反清领袖。
                    1937年10月，参加人民自卫军军政干部学校学习，后任**蠡**县六区农会宣传部长。
                    十一年，任直隶**蠡**县知县。
                    秦朝时，今肥城地属齐郡所辖，秦始皇东巡曾路过肥城，丞相李斯在范**蠡**墓前题词。

                    句子如下：
                    2013年12月28日，3号线工程在盘**蠡**路站举行动工仪式。

                    输出：
                    <think>
                    首先，我需要理解每个拼音的具体含义和用法。根据用户提供的信息，“蠡”在li2时用于表示一种容器，或是用户提供的例子比如“蠡王”，而在li3时更多用于动词或人名、地名中的特殊读音。例如，“范蠡”的“蠡”读作li3。
                    首先接下来，我要分析句子中的具体使用。“盘**蠡**路站”中的“蠡”显然是作为地名的一部分出现的。这里的“盘蠡路”可能是一个道路名称，类似于“盘锦路”或“盘龙路”。
                    接下来，我要理解每个拼音的具体含义和用法。根据提供的例子，“蠡”在“盘蠡路站”中作为地名的一部分，应该遵循当地的标准发音。如果“蠡县”中的“蠡”确实读作li3，那么“盘蠡路站”中的“蠡”也应该读作li3。
                    最后，按照要求，我只需要输出提供的选项中的正确读音，并且不要加音调。所以答案就是li3。
                    </think>
                    <answer>li3</answer>
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
def cul_acc(data):
    correct = 0
    total = len(data)
    for item in data:
        # 使用正則表達式提取 <answer>...</answer> 內的拼音
        match = re.search(r"<answer>(.*?)</answer>", item["predict"])
        predicted_pinyin = match.group(1) if match else None
        # 與 groundtruth 比對
        if predicted_pinyin == item["groundtruth"]:
            correct += 1
        else:
            print(f"句子: {item['sentence']}")
            print(f"預測: {predicted_pinyin}, 答案: {item['groundtruth']}")
    accuracy = correct / total if total > 0 else 0
    return accuracy

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
        
    
    # accuracy = calculate_accuracy(predicts,  groundtruths)
    # print(f"Accuracy: {accuracy:.2f}%")
    save_result(sentences, predicts, groundtruths)
    save_wrong_result(sentences, predicts,  groundtruths)
    
    with open('result/result.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    accuracy = cul_acc(data)
    print(f"Accuracy: {accuracy:.2%}")
    # bleu_score = calculate_bleu4(predicts,  groundtruths)
    # print(f"BLEU-4 Score: {bleu_score:.2f}%")
        
        
        
        