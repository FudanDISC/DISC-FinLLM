import json
import random
import re
import nltk


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_questions_and_text(input_string):
    # 使用问号来分割字符串为问题列表
    questions_list = input_string.split("？")

    # 第一个问题及之前的所有问题合并为一个字符串
    first_content = "？".join(questions_list[:-1]) + "？"

    # 最后一个问题后面的文本内容作为第二个内容
    second_content = questions_list[-1]

    return first_content, second_content


def _mixed_segmentation(in_str, rm_punc=False):
    in_str = in_str.lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：', '？', '！', '“', '”', '；',
               '’',
               '《', '》', '……', '·', '、', '「', '」', '（', '）', '－', '～', '『', '』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(u'[\u4e00-\u9fa5]', char) or char in sp_char:  # chinese utf-8 code: u4e00 - u9fa5
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char
    # handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)
    return segs_out


# remove punctuation
def _remove_punctuation(in_str):
    in_str = in_str.lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：', '？', '！', '“', '”', '；',
               '’',
               '《', '》', '……', '·', '、', '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def _find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax

def _compute_f1_score(reference: str, prediction: str) -> float:
    f1_scores = []
    prediction_segment = _mixed_segmentation(prediction, rm_punc=True)

    reference_segment = _mixed_segmentation(reference, rm_punc=True)
    lcs, lcs_len = _find_lcs(reference_segment, prediction_segment)
    if lcs_len == 0:
        f1_scores = 0
    else:
        precision = 1.0 * lcs_len / len(prediction_segment)
        recall = 1.0 * lcs_len / len(reference_segment)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores = f1
    return f1_scores


def compute(*args, **kwargs) -> float:
    """Compute the metrics.
    Args:
        We disallow the usage of positional arguments to prevent mistakes
        `predictions` (Optional list/array/tensor): predictions
        `references` (Optional list/array/tensor): references
        `**kwargs` (Optional other kwargs): will be forwared to the metrics
    Return:
        Dictionnary with the metrics if this metric is run on the main process (process_id == 0)
        None if the metric is not run on the main process (process_id != 0)
    """
    if args:
        raise ValueError("Please call `compute` using keyword arguments.")
    predictions = kwargs.pop("predictions", None)
    # print(predictions)
    references = kwargs.pop("references", None)
    # print(references)
    f1, em, total_count = 0, 0, 0
    for reference, prediction in zip(references, predictions):
        total_count += 1
        f1 += _compute_f1_score(reference, prediction)
        # em += _compute_em_score(reference, prediction)
        # print(f1)
    f1_score = 100.0 * f1 / total_count
    # em_score = 100.0 * em / total_count
    return f1_score

# def check_conditions(selected_lists):
#     conditions = set(["condition1", "condition2", "condition3"])  # 替换为实际的三种条件值
#     second_elements = set(item[1] for item in selected_lists)
#     return conditions.issubset(second_elements)
#
# def random_selection_with_conditions(big_list, k):
#     valid_lists = [item for item in big_list if item[1] in ["condition1", "condition2", "condition3"]]
#
#     if len(valid_lists) < k:
#         raise ValueError("Not enough valid lists to select from.")
#
#     selected_lists = random.sample(valid_lists, k)
#
#     while not check_conditions(selected_lists):
#         selected_lists = random.sample(valid_lists, k)
#
#     return selected_lists
