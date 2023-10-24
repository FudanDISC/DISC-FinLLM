import os
from tqdm import tqdm

from utils import _remove_punctuation, _mixed_segmentation, _find_lcs, write_json, load_json, extract_questions_and_text, _compute_f1_score, compute


DATA_PATH = 'data/fincuge'
INSTRUCT_SAMPLES = load_json('data/fincuge/instruct_samples.json')

# class Evaluator:
#     __metaclass__ = ABCMeta  # 必须先声明
#     def __init__(self,instance_1,instance_2,instance_3,instance_4,instance_5,instance_6a,instance_6b,instance_7,instance_8):
#         self.fqael = [instance_1,instance_2,instance_3,instance_4,instance_5,instance_6a,instance_6b,instance_7,instance_8]
#     def run(self):
#         eval_dict={}
#         for i,eval in tqdm(enumerate(self.fqael)):
#             r=eval.run_evaluation(llm)
#             eval_dict.update(r)
#             print(eval_dict)
#         return eval_dict
    
class FinFEEvaluator:

    dataset = 'finfe'

    zero_shot_prompts = [
        '请根据上下文，从股民论坛中提取股民评论所表达的情绪，选项为积极、消极、中性，请在这三个选项中选出唯一正确的选项。请仅输出情绪类别，多余文字不要输出。\n\n上下文：{context}\n选项：积极、消极、中性\n答案：',
        '下面是一段股民论坛中股民的评论，你可以告诉我该评论的情绪倾向是什么吗？积极、消极还是中性？你只需要回答情绪“积极”、“消极”或“中性”，不需要给出其他内容。\n\n上下文：{context}\n答案：',
        '上下文：{context}\n请根据上下文，选出此文本所表现出的情绪，选项为积极、消极、中性。只需要回答情绪“积极”、“消极”或“中性”，不需要给出其他内容。\n答案：'
    ]

    few_shot_prompts = [
        '请根据上下文，从股民论坛中提取股民评论所表达的情绪，选项为积极、消极、中性，请在这三个选项中选出唯一正确的选项。请遵循以下示例，仅输出情绪类别，多余文字不要输出。下面给出了一些样例，按照样例输出答案。\n{context}',
        '下面是一段股民论坛中股民的评论，你可以告诉我该评论的情绪倾向是什么吗？积极、消极还是中性？请参考下面的例子进行回答。\n{context}',
        '请根据上下文，选出此文本所表现出的情绪，选项为积极、消极、中性。只需要回答情绪“积极”、“消极”或“中性”，不需要给出其他内容。请参考下面的例子进行回答。\n{context}'
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    def build_few_shot_prompt(self, prompt, context: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{context}\n选项：积极、消极、中性\n答案：{answer}'.format(
                context=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{context}\n选项：积极、消极、中性\n答案：'.format(context=context)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds):
        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            # 如果需要处理文本，可以调用 clean 函数，例如 if gold == clean(pred):
            if gold == _remove_punctuation(pred):
                s += 1
        return round(s / len(golds) *100, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=example['input']))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], context=example['input'], k=3))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=example['input'])
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, context=example['input'], k=few_shot_k)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        return {'zero_shot_metrics_finfe' : zero_shot_metrics, 'few_shot_metrics_finfe' : few_shot_metrics, 'avg_zero_shot' : avg_zero_shot, 'avg_few_shot' : avg_few_shot}

class FinQAEvaluator:

    dataset = 'finqa'

    zero_shot_prompts = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。\n\n上下文: {inp}\n问题: {ins}\n答案：',
        '我需要从下面的文本中识别事件信息，你可以帮助我回答下面的阅读理解问题吗？\n\n上下文：{inp}\n问题：{ins}\n答案：',
        '上下文：{inp}\n问题：{ins}请根据此上下文及问题，回答答案。'
    ]

    few_shot_prompts = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
        '我需要从下面的文本中识别事件信息，你可以帮助我回答下面的阅读理解问题吗？你的回答可以参考下面的样例。\n\n{context}',
        '请根据提供的文本信息，回答问题。下面给出了一个例子：\n{context}'
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, inp, ins):
        return prompt.format(inp=inp, ins=ins)

    def build_few_shot_prompt(self, prompt, inp1: str, ins1: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            ins, inp = extract_questions_and_text(instruct['input'])
            instruct_prompts.append('上下文：{inp}\n问题：{ins}\n答案：{answer}'.format(
                ins=ins, inp=inp, answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n问题：{ins}\n答案：'.format(inp=inp1, ins=ins1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds):
        assert len(golds) == len(preds)
        f1, total_count = 0, 0
        for gold, pred in zip(golds, preds):
            pred = _mixed_segmentation(pred, rm_punc=True)
            gold = _mixed_segmentation(gold, rm_punc=True)
            lcs, lcs_len = _find_lcs(gold, pred)
            if lcs_len == 0:
                score = 0
            else:
                precision = 1.0 * lcs_len / len(pred)
                recall = 1.0 * lcs_len / len(gold)
                score = (2 * precision * recall) / (precision + recall)
            total_count += 1
            f1 += score
        f1_score = 100.0 * f1 / total_count

        return round(f1_score, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        ins, inp = extract_questions_and_text(example['input'])
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp, ins=ins))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        ins, inp = extract_questions_and_text(example['input'])
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, ins1=ins, k=1))

    def run_evaluation(self, llm, few_shot_k: int = 1):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                ins, inp = extract_questions_and_text(example['input'])
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp, ins=ins)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot  += self.evaluate(golds, preds)
        nums_zero_shot = len(self.few_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                ins, inp = extract_questions_and_text(example['input'])
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, ins1=ins, k=few_shot_k)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot  += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        return {'zero_shot_metrics_finqa': zero_shot_metrics, 'few_shot_metrics_finqa': few_shot_metrics, 'avg_zero_shot_finqa' : avg_zero_shot, 'avg_few_shot_finqa' : avg_few_shot}

class FinCQAEvaluator:

    dataset = 'fincqa'

    zero_shot_prompts = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。\n\n上下文: {inp}\n问题: {ins}\n答案：',
        '我需要从下面的文本中识别事件信息，你可以帮助我回答下面的阅读理解问题吗？\n\n上下文：{inp}\n问题：{ins}\n答案：',
        '上下文：{inp}\n问题：{ins}请根据此上下文及问题，回答答案。'
    ]

    few_shot_prompts = [
        '请从文本中识别事件信息，根据上下文及问题，以阅读理解问答的形式，回答问题的答案。下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
        '我需要从下面的文本中识别事件信息，你可以帮助我回答下面的阅读理解问题吗？你的回答可以参考下面的样例。\n\n{context}',
        '请根据提供的文本信息，回答问题。下面给出了一个例子：\n{context}'
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, inp, ins):
        return prompt.format(inp=inp, ins=ins)

    def build_few_shot_prompt(self, prompt, inp1: str, ins1: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            ins, inp = extract_questions_and_text(instruct['input'])
            instruct_prompts.append('上下文：{inp}\n问题：{ins}\n答案：{answer}'.format(
                ins=ins, inp=inp, answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n问题：{ins}\n答案：'.format(inp=inp1, ins=ins1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds):
        assert len(golds) == len(preds)
        f1, total_count = 0, 0
        for gold, pred in zip(golds, preds):
            pred = _mixed_segmentation(pred, rm_punc=True)
            gold = _mixed_segmentation(gold, rm_punc=True)
            lcs, lcs_len = _find_lcs(gold, pred)
            if lcs_len == 0:
                score = 0
            else:
                precision = 1.0 * lcs_len / len(pred)
                recall = 1.0 * lcs_len / len(gold)
                score = (2 * precision * recall) / (precision + recall)
            total_count += 1
            f1 += score
        f1_score = 100.0 * f1 / total_count

        return round(f1_score, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        ins, inp = extract_questions_and_text(example['input'])
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp, ins=ins))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        ins, inp = extract_questions_and_text(example['input'])
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, ins1=ins, k=1))

    def run_evaluation(self, llm, few_shot_k: int = 1):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                ins, inp = extract_questions_and_text(example['input'])
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp, ins=ins)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                ins, inp = extract_questions_and_text(example['input'])
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, ins1=ins, k=few_shot_k)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot  += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        return {'zero_shot_metrics_fincqa' : zero_shot_metrics, 'few_shot_metrics_fincqa' : few_shot_metrics, 'avg_zero_shot_fincqa' : avg_zero_shot, 'avg_few_shot_fincqa' : avg_few_shot}

class FinNAEvaluator:

    dataset = 'finna'

    zero_shot_prompts = [
        '请根据上下文给出的中文短新闻，生成对应的不超过20个字的摘要。\n\n上下文：{context}',
        '新闻：{context}\n你可以帮助我归纳一个不超过20字的摘要吗？',
        '上下文：{context}\n请根据上下文给出的新闻，生成对应的不超过20个字的简短摘要。'
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, context):
        return prompt.format(context=context)

    @staticmethod
    def evaluate(golds, preds):
        import jieba
        from rouge_chinese import Rouge
        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            string2 = pred
            string1 = gold
            # 创建 Rouge 对象
            rouge = Rouge()
            # 对字符串进行分词
            string1_tokens = jieba.cut(string1)
            string2_tokens = jieba.cut(string2)
            string1_text = " ".join(string1_tokens)
            string2_text = " ".join(string2_tokens)
            scores = rouge.get_scores(string1_text, string2_text)
            rouge_l_f1 = scores[0]['rouge-l']['f']
            s += rouge_l_f1
        num_all = len(golds)
        acc_rate = s / num_all *100
        return round(acc_rate, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], context=example['input']))

    def run_evaluation(self, llm):
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, context=example['input'])
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        return {'zero_shot_metrics_finfe' : zero_shot_metrics, 'avg_zero_shot_finna' : avg_zero_shot}

class FinREEvaluator:

    dataset = 'finre'

    zero_shot_prompts = [
        '请根据财经金融领域文本及问题，在给定的关系选项里进行问题答案的选择，仅选择唯一正确的一个答案，并直接输出答案，不要输出任何其他文字内容。\n\n可能的关系选项为：分析、借壳、合作、转让、买资、入股、商讨、被拟收购、被成立、重组、自己、被注资、被分析、被帮助、交易、被持股、被入股、注资、成立、被买资、被借壳、增持、被拥有、发行、订单、拥有、纠纷、被增持、被转让、合资、减持、欠款、其他、被减持、签约、拟收购、被收购、合并、帮助、被发行、被欠款、持股、收购、竞争\n请在上面这些类别里对两个主体的关系进行选择。如果没有特殊的关系，请输出其他。\n\n上下文：{inp}\n问题：{insa}和{insb}之间的关系是什么？\n答案：',
        '有一篇财经文本是这样描述的：“{inp}”\n\n请问在上面的文本中，{insa}和{insb}之间的关系属于哪一个类别？你可以选择的关系类别包括：分析、借壳、合作、转让、买资、入股、商讨、被拟收购、被成立、重组、自己、被注资、被分析、被帮助、交易、被持股、被入股、注资、成立、被买资、被借壳、增持、被拥有、发行、订单、拥有、纠纷、被增持、被转让、合资、减持、欠款、其他、被减持、签约、拟收购、被收购、合并、帮助、被发行、被欠款、持股、收购、竞争\n请注意，你只可以选择一个类别，如果{insa}和{insb}之间的关系不属于上面这些类别，你可以请输出其他。\n\n答案：',
        '请根据财经金融领域文本及问题，在给定的关系选项里进行问题答案的选择，仅选择唯一正确的一个答案，并直接输出答案，不要输出任何其他文字内容。\n\n请在下面这些类别里对两个主体的关系进行选择。如果没有特殊的关系，请输出其他。\n\n上下文：{inp}\n可能的关系选项为：分析、借壳、合作、转让、买资、入股、商讨、被拟收购、被成立、重组、自己、被注资、被分析、被帮助、交易、被持股、被入股、注资、成立、被买资、被借壳、增持、被拥有、发行、订单、拥有、纠纷、被增持、被转让、合资、减持、欠款、其他、被减持、签约、拟收购、被收购、合并、帮助、被发行、被欠款、持股、收购、竞争\n问题：{insa}和{insb}之间的关系是什么？\n答案：'
    ]

    few_shot_prompts = [
        '请根据财经金融领域文本及问题，在给定的关系选项里进行问题答案的选择，仅选择唯一正确的一个答案，并直接输出答案，不要输出任何其他文字内容。\n\n可能的关系选项为：分析、借壳、合作、转让、买资、入股、商讨、被拟收购、被成立、重组、自己、被注资、被分析、被帮助、交易、被持股、被入股、注资、成立、被买资、被借壳、增持、被拥有、发行、订单、拥有、纠纷、被增持、被转让、合资、减持、欠款、其他、被减持、签约、拟收购、被收购、合并、帮助、被发行、被欠款、持股、收购、竞争\n请在上面这些类别里对两个主体的关系进行选择。如果没有特殊的关系，请输出其他。下面给出示例：\n\n{context}',
        '请根据财经文本的文本，回答问题。你答案可以选择的关系类别包括：分析、借壳、合作、转让、买资、入股、商讨、被拟收购、被成立、重组、自己、被注资、被分析、被帮助、交易、被持股、被入股、注资、成立、被买资、被借壳、增持、被拥有、发行、订单、拥有、纠纷、被增持、被转让、合资、减持、欠款、其他、被减持、签约、拟收购、被收购、合并、帮助、被发行、被欠款、持股、收购、竞争\n请注意，你只可以选择一个类别，如果两者之间的关系不属于上面这些类别，你可以请输出其他。下面给出示例：n\n{context}'
        '参考材料，请问给出的两个主体的关系属于下面给出类别里的哪个？你答案可以选择的关系类别包括：分析、借壳、合作、转让、买资、入股、商讨、被拟收购、被成立、重组、自己、被注资、被分析、被帮助、交易、被持股、被入股、注资、成立、被买资、被借壳、增持、被拥有、发行、订单、拥有、纠纷、被增持、被转让、合资、减持、欠款、其他、被减持、签约、拟收购、被收购、合并、帮助、被发行、被欠款、持股、收购、竞争\n\n下面给出示例：n\n{context}'
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, inp, insa, insb):
        return prompt.format(inp=inp, insa=insa, insb=insb)

    def build_few_shot_prompt(self, prompt, inp1: str, insa1: str, insb1: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{inp}\n问题：{insa}和{insb}之间的关系是什么？\n答案:{answer}'.format(
                inp=instruct['input'][0], insa=instruct['input'][1], insb=instruct['input'][2], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n问题：{insa}和{insb}之间的关系是什么？\n答案：'.format(inp=inp1, insa=insa1, insb=insb1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds):
        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            # 如果需要处理文本，可以调用 clean 函数，例如 if gold == clean(pred):
            if gold == _remove_punctuation(pred):
                s += 1
        return round(s / len(golds) *100, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp, insa, insb = example['input'][0], example['input'][1], example['input'][2]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp, insa=insa, insb=insb))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp, insa, insb = example['input'][0], example['input'][1], example['input'][2]
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, insa1=insa, insb1=insb, k=1))

    def run_evaluation(self, llm, few_shot_k: int = 1):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                inp, insa, insb = example['input'][0], example['input'][1], example['input'][2]
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp, insa=insa, insb=insb)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                inp, insa, insb = example['input'][0], example['input'][1], example['input'][2]
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, insa1=insa, insb1=insb, k=few_shot_k)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        return {'zero_shot_metrics_finre': zero_shot_metrics, 'few_shot_metrics_finre': few_shot_metrics, 'avg_zero_shot_finre' : avg_zero_shot, 'avg_few_shot_finre' : avg_few_shot}

class FinNSP1Evaluator:

    dataset = 'finnsp1'

    zero_shot_prompts = [
        '请根据上下文，判定该文本是否包含金融实体的负面信息。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则输出0。如果包含金融实体的负面信息，则输出1。请仅输出0或1。\n\n上下文：{inp}\n答案：',
        '文本：{inp}\n上面的文本中是否包含某个金融实体的负面信息呢？如果没有上文没有包含某一个实体的负面信息，或是包含负面信息但是没有提到负面信息相关的实体，请回答0；如果既包含负面信息，又提到了涉及负面信息的金融实体，请回答1。请注意，你的回答只能为0或1。',
        '请根据下面给出的材料，判断出该文本是否包含金融实体的负面信息。如果包含金融实体的负面信息，则输出1。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则输出0。请仅输出0或1。\n\n上下文：{inp}\n答案：'
    ]

    few_shot_prompts = [
        '请根据上下文，判定该文本是否包含金融实体的负面信息。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则输出0。如果包含金融实体的负面信息，则输出1。请遵循以下示例，仅输出0或1。下面给出了几个样例，按照此样例输出最后一个的答案。\n{context}',
        '下面的文本中是否包含某个金融实体的负面信息呢？如果没有上文没有包含某一个实体的负面信息，或是包含负面信息但是没有提到负面信息相关的实体，请回答0；如果既包含负面信息，又提到了涉及负面信息的金融实体，请回答1。你可以参考下面的几个例子，并给出最后一个例子的答案。\n{context}',
        '请根据下面给出的材料，判断出该文本是否包含金融实体的负面信息。如果包含金融实体的负面信息，则输出1。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则输出0。请仅输出0或1。\n\n下面给出几个示例：\n{context}'
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, inp):
        return prompt.format(inp=inp)

    def build_few_shot_prompt(self, prompt, inp1: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{inp}\n答案：{answer}'.format(
                inp=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n答案：'.format(inp=inp1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds):
        assert len(golds) == len(preds)
        s = 0
        for gold, pred in zip(golds, preds):
            # 如果需要处理文本，可以调用 clean 函数，例如 if gold == clean(pred):
            if gold == _remove_punctuation(pred):
                s += 1
        return round(s / len(golds) *100, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp = example['input']
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp = example['input']
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, k=2))

    def run_evaluation(self, llm, few_shot_k: int = 2):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                inp = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                inp = example['input']
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, k=few_shot_k)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        return {'zero_shot_metrics_finnsp1': zero_shot_metrics, 'few_shot_metrics_finnsp1': few_shot_metrics, 'avg_zero_shot_finnsp1' : avg_zero_shot, 'avg_few_shot_finnsp1' : avg_few_shot}

class FinNSP2Evaluator:

    dataset = 'finnsp2'

    zero_shot_prompts = [
        '根据上下文，判断负面信息的主体对象是给定的实体列表中的哪些实体，输出这些选出的实体。如果是多个实体，用逗号进行分割。请直接输出实体，不要输出任何其他文字，并用逗号把多个实体进行分割。\n\n上下文：{inp}\n给定的实体列表：{ins}\n答案：',
        '下面的文本中包含了一些负面信息，同时还给出了一个实体列表，你能帮我抽取到负面信息的主体对象是什么吗？请注意，你的主体对象必须在实体列表中。你的回答只需要包含实体名称，如果要输出多个实体，请用逗号作分隔。\n\n上下文：{inp}\n给定的实体列表：{ins}\n答案：',
        '请参考给出的材料信息，在给定的实体列表中，选出并输出负面信息的主体对象是哪些实体。如果是多个实体，用逗号进行分割。请直接输出实体，不要输出任何其他文字，并用逗号把多个实体进行分割。\n\n材料：{inp}\n可选择的实体列表：{ins}\n答案：'
    ]

    few_shot_prompts = [
        '根据上下文，判断负面信息的主体对象是给定的实体列表中的哪些实体，输出这些选出的实体。如果是多个实体，用逗号进行分割。请直接输出实体，不要输出任何其他文字，并用逗号把多个实体进行分割。下面给出了一个样例，按照此样例输出最后一个的答案。\n{context}',
        '下面的文本中包含了一些负面信息，同时还给出了一个实体列表，你能帮我抽取到负面信息的主体对象是什么吗？请注意，你的主体对象必须在实体列表中。你的回答只需要包含实体名称，如果要输出多个实体，请用逗号作分隔。\n你可以参考下面的样例，然后你需要给出最后一个问题的答案。\n{context}',
        '请参考给出的材料信息，在给定的实体列表中，选出并输出负面信息的主体对象是哪些实体。如果是多个实体，用逗号进行分割。请直接输出实体，不要输出任何其他文字，并用逗号把多个实体进行分割。\n{context}'
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, inp, ins):
        return prompt.format(inp=inp, ins=ins)

    def build_few_shot_prompt(self, prompt, inp1: str, ins1: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{inp}\n给定的实体列表：{ins}\n答案：{answer}'.format(
                inp=instruct['input'][0], ins= instruct['input'][1], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n给定的实体列表：{ins}\n答案：'.format(inp=inp1, ins=ins1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds, sep):
        assert len(golds) == len(preds)
        n1, n2, n3 = 0, 0, 0
        for reference, prediction in zip(golds, preds):
            reference = reference.split(';')
            _prediction = prediction.split(sep)
            # n1表示预测对的实体总个数
            n1 += len(set(reference).intersection(set(_prediction)))
            # n2表示所有的实体数
            n2 += len(reference)
            # n3表示预测的实体数
            n3 += len(_prediction)
        p = n1 / n3
        r = n1 / n2
        if p + r != 0:
            f1 = 2 * ((p * r) / (p + r)) * 100
        else:
            f1 = 0
        return round(f1, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp, ins = example['input'][0], example['input'][1]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp, ins=ins))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp, ins = example['input'][0], example['input'][1]
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, ins1=ins, k=3))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                inp, ins = example['input'][0], example['input'][1]
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp, ins=ins)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds, sep='，'))
            all_zero_shot += self.evaluate(golds, preds, sep='，')
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                inp, ins = example['input'][0], example['input'][1]
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, ins1=ins, k=few_shot_k)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds, sep='，'))
            all_few_shot += self.evaluate(golds, preds, sep='，')
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        return {'zero_shot_metrics_finnsp2': zero_shot_metrics, 'few_shot_metrics_finnsp2': few_shot_metrics, 'avg_zero_shot_finnsp2' : avg_zero_shot, 'avg_few_shot_finnsp2' : avg_few_shot}

class FinNLEvaluator:

    dataset = 'finnl'

    zero_shot_prompts = [
        '请根据新浪财经的新闻，分析出与上下文内容描述相关的金融新闻类别，并只在给出的选择范围中进行选择2-3个，并将多个类别用逗号进行分割。请遵循以下指示：直接输出类别，以最精炼的形式，不需要输出原因及多余文字。\n可选择的范围：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n\n上下文：{inp}\n答案：',
        '下面给出了一则财经新闻，你可以帮我判断它属于哪一个类别吗？你可以选择的类别包括：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n\n请注意，你只可以选择上述类别中的2-3项，以逗号作分隔，并且你只需要输出类别名称，不要输出其他内容。\n文本：{inp}\n答案：',
        '参考下面给出的新浪财经的新闻，以及给出的可选择的新闻类别，选出2-3个与内容描述相关的金融新闻类别。\n请注意，你只可以选择上述类别中的2-3项，以逗号作分隔，并且你只需要输出类别名称，不要输出其他内容。\n\n上下文：{inp}\n可选择的范围：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n答案：'
    ]

    few_shot_prompts = [
        '请根据新浪财经的新闻，分析出与上下文内容描述相关的金融新闻类别，并只在给出的选择范围中进行选择2-3个，并将多个类别用逗号进行分割。请遵循以下指示：直接输出类别，以最精炼的形式，不需要输出原因及多余文字。下面给出了一个样例，按照此样例输出最后一个的答案。\n可选择的范围：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n{context}',
        '下面给出了一些财经新闻，你可以帮我判断它属于哪一个类别吗？你可以选择的类别包括：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n\n请注意，你只可以选择上述类别中的2-3项，以逗号作分隔，并且你只需要输出类别名称，不要输出其他内容。同时，你可以参考下面的样例，然后你需要给出最后一则文本的分类。\n{context}',
        '参考下面给出的新浪财经的新闻，以及给出的可选择的新闻类别：公司、行业、大盘、中国、外国、国际、经济、政策、期货、债券、房地产、外汇、虚拟货币、新冠、能源\n选出2-3个与内容描述相关的金融新闻类别。\n请注意，你只可以选择上述类别中的2-3项，以逗号作分隔，并且你只需要输出类别名称，不要输出其他内容。\n{context}'
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, inp):
        return prompt.format(inp=inp)

    def build_few_shot_prompt(self, prompt, inp1: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{inp}\n答案：{answer}'.format(
                inp=instruct['input'], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n答案：'.format(inp=inp1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds, sep):
        assert len(golds) == len(preds)
        n1, n2, n3 = 0, 0, 0
        for reference, prediction in zip(golds, preds):
            reference = reference.split(' ')
            _prediction = prediction.split(sep)
            # n1表示预测对的实体总个数
            n1 += len(set(reference).intersection(set(_prediction)))
            # n2表示所有的实体数
            n2 += len(reference)
            # n3表示预测的实体数
            n3 += len(_prediction)
        p = n1 / n3
        r = n1 / n2
        if p + r != 0:
            f1 = 2 * ((p * r) / (p + r)) * 100
        else:
            f1 = 0
        return round(f1, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp = example['input']
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp = example['input']
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, k=3))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                inp = example['input']
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds, sep='，'))
            all_zero_shot += self.evaluate(golds, preds, sep='，')
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                inp = example['input']
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, k=few_shot_k)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds, sep='，'))
            all_few_shot += self.evaluate(golds, preds, sep='，')
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        return {'zero_shot_metrics_finnl': zero_shot_metrics, 'few_shot_metrics_finnl': few_shot_metrics, 'avg_zero_shot_finnl' : avg_zero_shot, 'avg_few_shot_finnl' : avg_few_shot}

class FinESEEvaluator:

    dataset = 'finese'

    zero_shot_prompts = [
        '请根据给出上下文的新闻语料中，以及给定的事件类型，抽取在上下文中的此特定事件类型的主体。如果没抽取出此特定事件类型的主体，请输出"NaN"。\n\n上下文：{inp}\n事件类型：{ins}\n答案：',
        '下面有一则新闻文本，请问在文本中与“{ins}”事件有关的主体是什么？如果没有提到这一事件的主体，你可以回答"NaN"。\n文本：{inp}\n答案：',
        '参考给出的材料，以及给定的事件类型，判断出在此材料中的此特定事件类型的主体。如果没判断出此特定事件类型的主体，请输出"NaN"。\n\n材料：{inp}\n事件类型：{ins}\n答案：',

    ]

    few_shot_prompts = [
        '请根据给出上下文的新闻语料中，以及给定的事件类型，抽取在上下文中的此特定事件类型的主体。如果没抽取出此特定事件类型的主体，请输出"NaN"。下面给出了三个样例，按照此样例输出最后一个的答案。\n{context}',
        '下面有一些新闻文本，请问在文本中与事件有关的主体是什么？如果没有提到这一事件的主体，你可以回答"NaN"。同时，你可以参考下面的样例，然后你需要给出最后一则文本的分类。\n{context}',
        '参考给出的材料，以及给定的事件类型，判断出在此材料中的此特定事件类型的主体。如果没判断出此特定事件类型的主体，请输出"NaN"。下面给出一些示例：\n{context}'
    ]

    def __init__(self):

        self.data = load_json(os.path.join(DATA_PATH, self.dataset + '-eval.jsonl'))
        self.instructs = INSTRUCT_SAMPLES.get(self.dataset)

    @staticmethod
    def build_zero_shot_prompt(prompt, inp, ins):
        return prompt.format(inp=inp, ins=ins)

    def build_few_shot_prompt(self, prompt, inp1: str, ins1: str, k: int):
        # 基于给定的例子，构建few shot模板
        instruct_prompts = []
        for instruct in self.instructs[: k]:
            instruct_prompts.append('上下文：{inp}\n事件类型：{ins}\n答案：{answer}'.format(
                inp=instruct['input'][0], ins= instruct['input'][1], answer=instruct['gold_answer']))
        sample_prompt = '上下文：{inp}\n事件类型：{ins}\n答案：'.format(inp=inp1, ins=ins1)
        return prompt.format(context='\n\n'.join(instruct_prompts) + '\n\n' + sample_prompt)

    @staticmethod
    def evaluate(golds, preds):
        assert len(golds) == len(preds)
        f1, total_count = 0, 0
        for gold, pred in zip(golds, preds):
            pred = _mixed_segmentation(pred, rm_punc=True)
            gold = _mixed_segmentation(gold, rm_punc=True)
            lcs, lcs_len = _find_lcs(gold, pred)
            if lcs_len == 0:
                score = 0
            else:
                precision = 1.0 * lcs_len / len(pred)
                recall = 1.0 * lcs_len / len(gold)
                score = (2 * precision * recall) / (precision + recall)
            total_count += 1
            f1 += score
        f1_score = 100.0 * f1 / total_count

        return round(f1_score, 1)

    # 打印 zero shot 输入示例
    def show_zero_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp, ins = example['input'][0], example['input'][1]
        print('-' * 50)
        print(self.build_zero_shot_prompt(prompt=self.zero_shot_prompts[j], inp=inp, ins=ins))

    # 打印 few shot 输入示例
    def show_few_shot_prompt(self, i=0, j=0):
        example = self.data[i]
        inp, ins = example['input'][0], example['input'][1]
        print('-' * 50)
        print(self.build_few_shot_prompt(prompt=self.few_shot_prompts[j], inp1=inp, ins1=ins, k=3))

    def run_evaluation(self, llm, few_shot_k: int = 3):
        # 先跑 zero shot
        all_zero_shot = 0
        zero_shot_metrics = []
        for zero_shot_prompt in self.zero_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                inp, ins = example['input'][0], example['input'][1]
                input_text = self.build_zero_shot_prompt(prompt=zero_shot_prompt, inp=inp, ins=ins)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            zero_shot_metrics.append(self.evaluate(golds, preds))
            all_zero_shot += self.evaluate(golds, preds)
        nums_zero_shot = len(self.zero_shot_prompts)
        avg_zero_shot = all_zero_shot / nums_zero_shot
        # 再跑 few shot
        all_few_shot = 0
        few_shot_metrics = []
        for few_shot_prompt in self.few_shot_prompts:
            golds, preds = [], []
            for example in tqdm(self.data):
                inp, ins = example['input'][0], example['input'][1]
                input_text = self.build_few_shot_prompt(prompt=few_shot_prompt, inp1=inp, ins1=ins, k=few_shot_k)
                preds.append(llm.generate(input_text))
                golds.append(example['gold_answer'])
            few_shot_metrics.append(self.evaluate(golds, preds))
            all_few_shot += self.evaluate(golds, preds)
        nums_few_shot = len(self.few_shot_prompts)
        avg_few_shot = all_few_shot / nums_few_shot
        return {'zero_shot_metrics_finese': zero_shot_metrics, 'few_shot_metrics_finese': few_shot_metrics, 'avg_zero_shot_finese' : avg_zero_shot, 'avg_few_shot_finese' : avg_few_shot}

if __name__ == '__main__':
    
    pass

    # evaluator = FinFEEvaluator()
    # evaluator.show_zero_shot_prompt()
    # evaluator.show_few_shot_prompt()
    # model = finllm.model
    # lora_path = finllm.lora_path
    # if model == 'baichuan7b':
    #     llm = DISCVFINLLMBaichuan7B(lora_path)
    # elif model == 'bloomz7b':
    #     llm = DISCVFINLLMBloomz7B(lora_path)
    # elif model == 'baichuan13bbase':
    #     llm = DISCVFINLLMBaichuan13BBase(lora_path)
    # elif model == 'baichuan13bchat':
    #     llm = DISCVFINLLMBaichuan13BChat(lora_path)
    # elif model == 'fingpt':
    #     llm = FinGPTv3()
    # elif model == 'chatglm2':
    #     llm = DISCVFINLLMChatGLM26B(lora_path)
    # elif model == 'chatglm':
    #     llm = DISCVFINLLMChatGLM6B(lora_path)

    # # a1 = DISCVFINLLMBloomz7B()
    # # a1 = DISCVFINLLMBloomz7B('/remote-home/qswang/LLaMA-Efficient-Tuning/output/Bloomz7B')
    # # a1=DISCVFINLLMBaichuan7B()
    # # a1=DISCVFINLLMBaichuan7B('/remote-home/qswang/LLaMA-Efficient-Tuning/output/Baichuan7B')
    # # a1=DISCVFINLLMBaichuan13BBase()
    # # a1=DISCVFINLLMBaichuan13BBase('/remote-home/qswang/LLaMA-Efficient-Tuning/output/Baichuan7B')
    # i1 = FinFEEvaluator()
    # i2 = FinQAEvaluator()
    # i3 = FinCQAEvaluator()
    # i4 = FinNAEvaluator()
    # i5 = FinREEvaluator()
    # i6a = FinNSP1Evaluator()
    # i6b = FinNSP2Evaluator()
    # i7 = FinNLEvaluator()
    # i8 = FinESEEvaluator()

    # c = Evaluator(i1, i2, i3, i4, i5, i6a, i6b, i7, i8)
    # print('模型加载完！')

    # result = c.run()

    # print(result)
    # if model == 'baichuan7b' and lora_path == None:
    #     with open('output/baichuan7b1.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'baichuan7b' and lora_path != None:
    #     with open('output/baichuan7b2.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'baichuan13bbase' and lora_path == None:
    #     with open('output/baichuan13b1.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'baichuan13bbase' and lora_path != None:
    #     with open('output/baichuan13b2.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'bloomz7b' and lora_path == None:
    #     with open('output/bloomz7b1.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'bloomz7b' and lora_path != None:
    #     with open('output/bloomz7b2.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'baichuan13bchat' and lora_path != None:
    #     with open('output/baichuan13bchat_1019.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'baichuan13bchat' and lora_path == None:
    #     with open('output/baichuan13bchat_1015.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'fingpt':
    #     with open('output/fingpt.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'chatglm2' and lora_path == None:
    #     with open('output/chatglm21.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'chatglm2' and lora_path != None:
    #     with open('output/chatglm22.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'chatglm' and lora_path == None:
    #     with open('output/chatglm11.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)
    # elif model == 'chatglm' and lora_path != None:
    #     with open('output/chatglm12.json', 'w') as f:
    #         json.dump(result, f, ensure_ascii=False, indent=4)

