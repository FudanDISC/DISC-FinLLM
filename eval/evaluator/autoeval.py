import argparse

from evaluate import *
from finllm import *
from utils import *


model_lists = {
    'chatglm-6b': DISCVFINLLMChatGLM6B,
    'chatglm2-6b': DISCVFINLLMChatGLM26B,
    'baichuan-7b': DISCVFINLLMBaichuan7B,
    'baichuan-13b-base': DISCVFINLLMBaichuan13BBase,
    'baichuan-13b-chat': DISCVFINLLMBaichuan13BChat,
    'bloomz-7b': DISCVFINLLMBloomz7B,
    'fingpt-v3': FinGPTv3,
}

Eval_datasets = {
    'finfe': FinFEEvaluator,  # 情感分析，非生成类任务，多分类
    'finqa': FinQAEvaluator,  # 问答，生成类任务
    'fincqa': FinCQAEvaluator,  # 问答，生成类任务
    'finna': FinNAEvaluator,  # 摘要，生成类任务
    'finre': FinREEvaluator, #关系分类，非生成类任务，多标签分类
    'finnsp1': FinNSP1Evaluator,  # 负面消息识别，非生成类任务
    'finnsp2': FinNSP2Evaluator,  # 负面主体判定，非生成类任务，多标签分类
    'finnl': FinNLEvaluator,  # 新闻分类，非生成类任务，多标签分类
    'finese': FinESEEvaluator,  # 事件主体抽取，生成类任务
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLM种类')
    parser.add_argument('--lora_path', default='', type=str, help='LORA路径')
    parser.add_argument('--eval_data', default='all', type=str, help='评测数据集名称')
    parser.add_argument('--device', default='cuda:0', type=str, help='推理的GPU设备')
    args = parser.parse_args()
    device = args.device
    model_name = args.model
    lora_path = None if args.lora_path == '' else args.lora_path
    eval_data = args.eval_data
    
    print(device)
    # 加载模型
    llm = model_lists.get(model_name)(device, lora_path)
    
    result_list = []
    if eval_data != 'all':
        assert eval_data in Eval_datasets
        evaluator = Eval_datasets.get(eval_data)
        evaluator().run_evaluation(llm)
    else:
        for _, evaluator in Eval_datasets.items():
            result = evaluator().run_evaluation(llm)
            result_list.append(result)
    if lora_path is None:
        write_json(f'{model_name}_eval.json',result_list)
    else:
        write_json(f'{model_name}_lora_eval.json',result_list)

    # usage:
    # python autoeval.py --model chatglm-6b --lora_path xxxx --eval_data all --device cuda:0

