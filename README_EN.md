<div align="center">
<h1>
  XVERSE-MoE-A36B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse">🤗 Hugging Face</a>&nbsp｜
        <a href="https://modelscope.cn/organization/xverse" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜
        <a href="resources/wechat.png">💬 WeChat</a>
</p>

<h4 align="left">
    <p>
        <a href="README.md">中文</a> |
        <b>English</b>
    <p>
</h4>

## Update Information
- **[2024/09/13]** Released **XVERSE-MoE-A36B** MoE base model, the Chat version model will be released later.

## Model Introduction

**XVERSE-MoE-A36B** is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology which is using Mixture-of-experts (MoE) architecture. The total parameter scale of the model is 255 billion, with an actual number of activated parameters being 36 billion. The models released this time is the base model **XVERSE-MoE-A36B**. Its key features are as follows:

- **Model Structure**: XVERSE-MoE-A36B uses the mainstream Decoder-only Transformer network structure that extends the FFN layer of dense models to expert layers. Unlike traditional MoE model where each expert has the same size as standard FFN (such as Mixtral 8x7B), it uses more fine-grained experts, with each expert being 1/4 the size of a standard FFN. It includes shared experts and non-shared experts, where shared experts are always activated during computation, and non-shared experts are selectively activated through a Router.
- **Training Data**: The model has been thoroughly trained on a large-scale high-quality dataset, including more than 40 languages such as Chinese, English, Russian, and Spanish. The sampling ratio of different types of data is finely set, which makes the performance of Chinese and English excellent, and also takes into account the effect of other languages; The model is trained using training samples of length 8k; During the model training process, several data switches were made to dynamically introduce continuously processed high-quality data, along with adjustments to the data sampling ratio.
- **Training Strategy**: While switching data, corresponding adjustments were also made to the learning rate scheduler to ensure the model could quickly and thoroughly learn from the newly introduced data.
- **Training Framework**: We conducted in-depth customized optimization for the unique expert routing and weight calculation logic in the MoE model, developed an efficient fusion operator to improve computational efficiency. At the same time, to address the challenges of high memory consumption and communication volume in the MoE model, we designed a processing method for overlapping computation, communication, and CPU-Offload to increase overall throughput.

The models sizes, architectures and learning rate of **XVERSE-MoE-A36B** are showed as follows:

| total params | activated params | n_layers | d_model | n_heads | d_ff | n_non_shared_experts | n_shared_experts | top_k |   lr   |
| :----------: | :--------------: | :------: | :-----: | :-----: | :--: | :------------------: | :--------------: | :---: | :----: |
|    255.4B    |       36.5B      |    50    |  6144   |   48    | 4096 |          64          |        2         |   6   | 2.5e−4 |

## Model Evaluation

To comprehensively assess the performance of the model, we conducted extensive testing across a range of standard datasets, including MMLU, C-Eval, CMMLU, RACE-M, PIQA, GSM8K, Math, MBPP and HumanEval. And compared it with open-weight MoE and Dense models (Base) of similar parameter scales, as well as closed-source Chat models. The results are as follows:

**Comparison of Open-Weight Base Models - MoE**
|              | XVERSE-MoE-A36B | Grok-1-A85B | DeepSeek-V2-A21B | Skywork-MoE-A22B | Mixtral-8x22B-A39B | DBRX-A36B |
| :----------: | :-------------: | :---------: | :--------------: | :--------------: | :----------------: | :-------: |
| Total Params |      255B       |    314B     |       236B       |       146B       |        141B        |   132B    |
|     MMLU     |    **80.8**     |     73      |       78.5       |       77.4       |        77.8        |   73.7    |
|    C-Eval    |      79.5       |      -      |       81.7       |       82.2       |        56.8        |   44.9    |
|    CMMLU     |      81.7       |      -      |        84        |       79.5       |        59.9        |   61.3    |
|    GSM8K     |    **89.5**     |    62.9     |       79.2       |       76.1       |        82.3        |   70.7    |
|     MATH     |    **53.3**     |    23.9     |       43.6       |       31.9       |        34.1        |   25.6    |
|  HumanEval   |      51.8       |    63.2     |       48.8       |       43.9       |        45.1        |   46.3    |
|     MBPP     |      59.8       |      -      |       66.6       |        -         |        71.2        |    58     |
|     PIQA     |    **84.8**     |      -      |       83.7       |        -         |        84.1        |   84.5    |
|    RACE-M    |    **88.4**     |      -      |       73.1       |        -         |        85.7        |   55.9    |

**Comparison of Open-Weight Base Models - Dense**
|              | XVERSE-MoE-A36B | XVERSE-65B-2 | Llama3.1-405B | Nemotron-4-340B | Qwen1.5-110B | Qwen2-72B | Qwen1.5-72B | Llama3.1-70B |
| :----------: | :-------------: | :----------: | :-----------: | :-------------: | :----------: | :-------: | :---------: | :----------: |
| Total Params |      255B       |     65B      |     405B      |      340B       |     110B     |    72B    |     72B     |     70B      |
|     MMLU     |      80.8       |     74.4     |     85.2      |      81.1       |     80.4     |   84.2    |    77.5     |     79.3     |
|    C-Eval    |      79.5       |     72.4     |       -       |        -        |     89.1     |    91     |    84.1     |      -       |
|    CMMLU     |      81.7       |     75.1     |       -       |        -        |     88.3     |   90.1    |    83.5     |      -       |
|    GSM8K     |    **89.5**     |     72.6     |      89       |        -        |     85.4     |   89.5    |    79.5     |     83.7     |
|     MATH     |      53.3       |     20.8     |     53.8      |        -        |     49.6     |   51.1    |    34.1     |     41.4     |
|  HumanEval   |      51.8       |     37.8     |      61       |      57.3       |     54.3     |   64.6    |    46.3     |     58.5     |
|     MBPP     |      59.8       |     40.6     |     73.4      |        -        |     70.9     |   76.9    |    66.9     |     66.2     |
|     PIQA     |      84.8       |     79.4     |     85.6      |        -        |              |     -     |      -      |     83.8     |
|    RACE-M    |      88.4       |     90.7     |       -       |        -        |              |     -     |      -      |      -       |

**Comparison of Closed-Source Chat Models**
|              | XVERSE-MoE-A36B | GPT-4o | abab-6.5-20240415 |     Step-2     |       Baichuan3       | GLM-4 (0520) |
| :----------: | :-------------: | :----: | :---------------: | :------------: | :-------------------: | :----------: |
| Total Params |      255B       |   -    |  Trillion scale   | Trillion scale | Hundred billion scale |      -       |
|     MMLU     |      80.8       |  88.7  |       78.7        |                |         81.7          |     83.3     |
|    C-Eval    |      79.5       |   -    |         -         |       -        |           -           |      -       |
|    CMMLU     |      81.7       |   -    |         -         |       -        |         78.1          |      -       |
|    GSM8K     |      89.5       |   -    |       91.7        |       94       |         88.2          |     93.3     |
|     MATH     |      53.3       |  76.6  |       51.3        |      68.4      |         49.2          |     61.3     |
|  HumanEval   |      51.8       |  90.2  |        78         |      84.1      |         70.1          |     78.5     |
|     MBPP     |      59.8       |   -    |         -         |       -        |         68.2          |      -       |
|     PIQA     |      84.8       |   -    |         -         |       -        |           -           |      -       |
|    RACE-M    |      88.4       |   -    |         -         |       -        |           -           |      -       |

For all the comparison models mentioned above, we report the maximum value between their official results and our self-evaluation results.

## Usage

### Environment Setup

1. Clone this repository:

```shell
git clone https://github.com/xverse-ai/XVERSE-MoE-A36B
cd XVERSE-MoE-A36B
```

2. Install the dependencies using pip:

```shell
pip install -r requirements.txt
```

### Loading with Transformers

The XVERSE-MoE-A36B model can be loaded for inference using the following code:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("xverse/XVERSE-MoE-A36B")
model = AutoModelForCausalLM.from_pretrained("xverse/XVERSE-MoE-A36B", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()
inputs = tokenizer('北京的景点：故宫、天坛、万里长城等。\n深圳的景点：', return_tensors='pt').input_ids
inputs = inputs.cuda()
generated_ids = model.generate(inputs, max_new_tokens=70, eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.1)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```

### Web Demo

The following code can be used to start a web server. By entering the access address in the browser, you can perform inference with the XVERSE-MoE-A36B model:

```shell
python chat_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

## Limitations and Disclaimer

Like all other Large Language Models (LLMs), XVERSE-MoE-A36B may produce inaccurate, biased, or otherwise offensive content under certain circumstances. Therefore, please use the content generated by the model with caution and refrain from disseminating harmful content. Before deploying any application of XVERSE-MoE-A36B, developers should conduct safety tests and optimization of the model according to its specific application.

We strongly warn against the use of the XVERSE-MoE-A36B model for producing or spreading harmful information, or conducting any activities that might harm the public, national, or social security, or violate regulations. We assume no responsibility for any problems arising from the use of the XVERSE-MoE-A36B model, whether it be data security issues, public opinion risks, or any risks and issues caused by misunderstanding, misuse, dissemination, or non-compliance with the model.

## Open Source License

The use of the source code in this repository must follow the [Apache-2.0](LICENSE) open-source license, while the use of the model weights of XVERSE-MoE-A36B needs to adhere to the [Model License Agreement](MODEL_LICENSE.pdf).

The XVERSE-MoE-A36B model weights are **fully open** to academic research and support **free commercial use**.  To apply for a commercial license, please fill in the [application form](https://chat.xverse.cn/home/business.html). For other questions or collaborations, please contact <opensource@xverse.cn>.

