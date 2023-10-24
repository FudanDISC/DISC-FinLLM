from abc import ABCMeta, abstractmethod
import re

from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import torch
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

class DISCFINLLMBase(metaclass=ABCMeta):

    @abstractmethod
    def generate(self, device: str, prompt: str) -> str:
        # 模型需要接收提示prompt，使用模型生成回复
        raise NotImplementedError


class DISCVFINLLMChatGLM26B(DISCFINLLMBase):
    def __init__(self, device: str = None, lora_path: str = None):
        model_name_or_path = "THUDM/chatglm2-6b"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(device)  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()

    def generate(self, prompt: str) -> str:
        answer, history = self.model.chat(self.tokenizer, prompt, history=[])
        return answer


class DISCVFINLLMChatGLM6B(DISCFINLLMBase):
    def __init__(self, device: str = None, lora_path: str = None):
        model_name_or_path = "THUDM/ChatGLM-6B"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path,
                                               trust_remote_code=True,
                                               torch_dtype=dtype).to(device)  # .half().cuda()
        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.model = self.model.eval()

    def generate(self, prompt: str) -> str:
        answer, history = self.model.chat(self.tokenizer, prompt, history=[])
        return answer


class DISCVFINLLMBaichuan13BBase(DISCFINLLMBase):
    def __init__(self,  device: str = None, lora_path: str = None):
        model_name_or_path = "baichuan-inc/Baichuan-13B-Base"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True).to(device)
        self.model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan-13B-Base")

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.device = device

    def generate(self, prompt: str) -> str:
        template = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "Human: {}\nAssistant: "
        )

        inputs = self.tokenizer([template.format(prompt)], return_tensors="pt")
        inputs = inputs.to(self.device)
        generate_ids = self.model.generate(**inputs, max_new_tokens=256)

        return generate_ids


class DISCVFINLLMBaichuan13BChat(DISCFINLLMBase):
    def __init__(self,  device: str = None, lora_path: str = None):
        model_name_or_path = "baichuan-inc/Baichuan-13B-Chat"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True).to(device)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
            print('lora加载完！')

    def generate(self, prompt: str) -> str:
        messages = []
        messages.append({"role": "user", "content": prompt})
        response = self.model.chat(self.tokenizer, messages)

        return response


class DISCVFINLLMBaichuan7B(DISCFINLLMBase):
    def __init__(self,  device: str = None, lora_path: str = None):
        model_name_or_path = "baichuan-inc/Baichuan-7B"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).half()
        self.model = self.model.to(device)

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)

        self.device = device
        
    def generate(self, prompt: str) -> str:
        template = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "Human: {}\nAssistant: "
        )

        inputs = self.tokenizer(template.format(prompt), return_tensors='pt')
        inputs = inputs.to(self.device)
        pred = self.model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
        answer = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        print(answer)
        pattern = answer.split('Assistant: ', 1)

        assistant_text = pattern[-1]
        print(assistant_text)
        return assistant_text


class DISCVFINLLMBloomz7B(DISCFINLLMBase):
    def __init__(self,  device: str = None, lora_path: str = None):
        model_name_or_path = "bigscience/bloomz-7b1-mt"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).half().to(device)

        if lora_path:
            peft_model_id = lora_path

            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
        self.device = device
        
    def generate(self, prompt: str) -> str:

        template = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "Human: {}\nAssistant: "
        )
        inputs = self.tokenizer.encode_plus(template.format(prompt), return_tensors='pt')
        outputs = self.model.generate(**inputs.to(self.device), max_new_tokens=128, repetition_penalty=1.1)
        answer = self.tokenizer.decode(outputs[0])
        pattern = r'Assistant: (.+?)(?:</s>|$)'
        matches = re.findall(pattern, answer)
        # 输出结果
        if matches != []:
            assistant_text = matches[0]
        else:
            assistant_text = '无'

        return assistant_text


class FinGPTv3:
    def __init__(self, device: str = None):
        model_name_or_path = "THUDM/chatglm2-6b"
        peft_model = "oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT"
        dtype = torch.float16
        # 训练后的lora保存的路径

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
        self.model = PeftModel.from_pretrained(self.model, peft_model)
        self.device =  device
        
    def generate(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt, return_tensors='pt', padding=True, max_length=512)
        res = self.model.generate(**tokens.to(self.device), max_length=512)
        res_sentences = self.tokenizer.decode(res[0])
        answer = res_sentences.replace(prompt, '').strip()
        
        return answer


if __name__ == '__main__':
    pass

