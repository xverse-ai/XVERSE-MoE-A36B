<div align="center">
<h1>
  XVERSE-MoE-A36B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse">🤗 Hugging Face</a>&nbsp｜
        <a href="https://modelscope.cn/organization/xverse" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜
        <a href="resources/wechat.png">💬 微信社区</a>
</p>

<h4 align="left">
    <p>
        <b>中文</b> |
        <a href="README_EN.md">English</a>
    <p>
</h4>

## 更新信息
- **[2024/09/13]** 发布 MoE 架构的 **XVERSE-MoE-A36B** 底座模型，Chat 对齐模型将在后续发布。

## 模型介绍

**XVERSE-MoE-A36B** 是由深圳元象科技自主研发的支持多语言的大语言模型（Large Language Model），使用混合专家模型（MoE，Mixture-of-experts）架构，模型的总参数规模为 2554 亿，实际激活的参数量为 360 亿，本次开源的模型为底座模型 **XVERSE-MoE-A36B**，主要特点如下：

- **模型结构**：XVERSE-MoE-A36B 为 Decoder-only 的 Transformer 架构，将密集模型的 FFN 层扩展为专家层，不同于传统 MoE 中每个专家的大小与标准 FFN 相同（如Mixtral 8x7B ），使用了更细粒度的专家，每个专家是标准 FFN 大小的 1/4，并设置了共享专家（Shared Expert）和非共享专家（Non-shared Expert）两类，共享专家在计算时始终被激活，非共享专家通过 Router 选择性激活。
- **训练数据**：构建了海量高质量、多样化的数据对模型进行充分训练，包含中、英、俄、西等 40 多种语言，通过精细化设置不同类型数据的采样比例，使得中英两种语言表现优异，也能兼顾其他语言效果；模型使用 8K 长度的训练样本进行训练；在模型训练过程中进行了若干次数据的切换，来动态的引入持续处理的高质量数据，同时伴随数据采样比的调整。
- **训练策略**：在切换数据的同时，为了使模型对新进数据进行快速且充分的学习，对学习率调度器也进行了相应调整。
- **训练框架**：针对 MoE 模型中独有的专家路由和权重计算逻辑，进行了深入定制优化，开发出一套高效的融合算子，以提升计算效率。同时，为解决 MoE 模型显存占用和通信量大的挑战，设计了计算、通信和 CPU-Offload 的 Overlap 处理方式，从而提高整体吞吐量。

**XVERSE-MoE-A36B** 的模型大小、架构和学习率如下：

| total params | activated params | n_layers | d_model | n_heads | d_ff | n_non_shared_experts | n_shared_experts | top_k |   lr   |
| :----------: | :--------------: | :------: | :-----: | :-----: | :--: | :------------------: | :--------------: | :---: | :----: |
|    255.4B    |       36.5B      |    50    |  6144   |   48    | 4096 |          64          |        2         |   6   | 2.5e−4 |

## 评测结果

为了综合评估模型的性能，我们在一系列标准数据集上进行了全面测试，包括 MMLU、C-Eval、CMMLU、RACE-M、PIQA、GSM8K、MATH、MBPP 和 HumanEval，这些评估数据集覆盖了模型在多个领域的能力。并与相近参数规模的开源 MoE 和 Dense 模型（Base）以及闭源 Chat 模型进行了对比，结果如下：

**对比开源 Base 模型 - MoE**
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

**对比开源 Base 模型 - Dense**
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

**对比闭源 Chat 模型**
|              | XVERSE-MoE-A36B | GPT-4o | abab-6.5-20240415 | Step-2 | Baichuan3 | GLM-4 (0520) |
| :----------: | :-------------: | :----: | :---------------: | :----: | :-------: | :----------: |
| Total Params |      255B       |   -    |       万亿        |  万亿  |   千亿    |      -       |
|     MMLU     |      80.8       |  88.7  |       78.7        |        |   81.7    |     83.3     |
|    C-Eval    |      79.5       |   -    |         -         |   -    |     -     |      -       |
|    CMMLU     |      81.7       |   -    |         -         |   -    |   78.1    |      -       |
|    GSM8K     |      89.5       |   -    |       91.7        |   94   |   88.2    |     93.3     |
|     MATH     |      53.3       |  76.6  |       51.3        |  68.4  |   49.2    |     61.3     |
|  HumanEval   |      51.8       |  90.2  |        78         |  84.1  |   70.1    |     78.5     |
|     MBPP     |      59.8       |   -    |         -         |   -    |   68.2    |      -       |
|     PIQA     |      84.8       |   -    |         -         |   -    |     -     |      -       |
|    RACE-M    |      88.4       |   -    |         -         |   -    |     -     |      -       |

对于上述所有比较模型，我们汇报其官方结果与自测结果之间的最大值。

## 使用方法

### 环境安装

1. 下载本仓库：

```shell
git clone https://github.com/xverse-ai/XVERSE-MoE-A36B
cd XVERSE-MoE-A36B
```

2. 使用 pip 安装依赖：

```shell
pip install -r requirements.txt
```
### Transformers 加载方式

可通过以下代码加载 XVERSE-MoE-A36B 模型来进行推理：

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

### 网页 Demo

可通过以下代码启动一个web server，在浏览器输入访问地址后，可使用 XVERSE-MoE-A36B 模型进行推理：

```shell
python text_generation_demo.py --port='port' --model_path='/path/to/model/' --tokenizer_path='/path/to/tokenizer/'
```

## 局限性与免责申明

XVERSE-MoE-A36B 与其他所有 LLM 一样，在某些情况下可能会产生不准确、有偏见或其他令人反感的内容。因此，请谨慎使用模型生成的内容，请勿将生成的有害内容进行传播，在部署任何 XVERSE-MoE-A36B 的应用之前，开发人员应根据其具体应用对模型进行安全测试和调优。

我们强烈警告不要将 XVERSE-MoE-A36B 模型用于制造或传播有害信息，或进行任何可能损害公众、国家、社会安全或违反法规的活动。如果使用 XVERSE-MoE-A36B 模型产生任何问题，无论是数据安全问题、公共舆论风险，还是模型被误解、滥用、传播或不合规使用所引发的任何风险和问题，我们将不承担任何责任。

## 模型开源协议

使用本仓库的源码需要遵循 [Apache-2.0](LICENSE) 开源协议，使用 XVERSE-MoE-A36B 的模型权重则需要遵循[模型许可协议](MODEL_LICENSE.pdf)。

XVERSE-MoE-A36B 模型权重对学术研究**完全开放**，并且支持**免费商用**。如需申请商业许可证，请填写【[申请表](https://chat.xverse.cn/home/business.html)】，如有其他问题或合作，请联系 <opensource@xverse.cn>。

