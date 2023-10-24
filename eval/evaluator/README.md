### 评测方法

首先，我们所用的评测数据集是超对称团队针对金融行业的预训练模型发布的一套目前中文领域金融大模型最专业的评测数据集BBT CFLEB，包含八个标准语言任务，包括摘要生成、文本分类、关系抽取、事件抽取和其他任务，用以衡量不同的模型的多维能力，并促进金融大模型研发。数据集出处：https://bbt.ssymmetry.com/evaluation.html。

我们目前提供了baichuan-7b、baichuan-13b-base、baichuan-13b-chat、bloomz-7b、chatglm、chatglm2、fingpt-v3的评测代码，同时也可以加入针对自己数据训练出来的lora权重进行评测。下面介绍评测的几个步骤：

- 第一步：运行脚本`preprocess.py`，会在本地创建一个`data`的文件夹，把八个任务的数据集下载到本地，并且会针对每个数据集进行相应的处理，方便后续进行评测。同时还会生成一个`instruct_samples.json`文件，这里保存着每个数据集的few-shot。这里需要说明的一点是：我们将第七个数据集拆分成两个数据集，对应着两个不同的任务。

- 第二步：使用如下命令运行脚本`autoeval.py`：
python autoeval.py --model xxxx --lora_path xxxx --eval_data all --device cuda:0

目前model可以分别设置为：chatglm-6b、chatglm2-6b、baichuan-7b、baichuan-13b-base、baichuan-13b-chat、bloomz-7b、fingpt-v3。

- 第三步：最终的评测结果会自动保存成json文件。


如果需要在其他模型上进行评测，需要修改源代码。

主要分为以下两步：

- 第一步：在 `finllm.py` 代码中自定义一个模型类，该类需要继承 DISCFINLLMBase 类，并实现 generate 函数，其中 generate 函数的输入为任意**提示文本**，输出为模型的**回复**

```python
import os
from evaluator.finllm import DISCFINLLMBase
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = ""


class OpenAILLM(DISCFINLLMBase):

    def __init__(self):
        self.model = ChatOpenAI()
    
    def generate(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.model(messages).content
        return response
```

- 第二步：运行脚本

```shell
python evaluate.py 
```

