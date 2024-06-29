# gptpdf

<p align="center">
<a href="README_CN.md"><img src="https://img.shields.io/badge/文档-中文版-blue.svg" alt="CN doc"></a>
<a href="README.md"><img src="https://img.shields.io/badge/document-English-blue.svg" alt="EN doc"></a>
</p>

使用视觉大语言模型（如 GPT-4o）将 PDF 解析为 markdown。

我们的方法非常简单(只有293行代码)，但几乎可以完美地解析排版、数学公式、表格、图片、图表等。

每页平均价格：0.013 美元

我们使用 [GeneralAgent](https://github.com/CosmosShadow/GeneralAgent) lib 与 OpenAI API 交互。



## 处理流程

1. 使用 PyMuPDF 库，对 PDF 进行解析出所有非文本区域，并做好标记，比如:

![](docs/demo.jpg)

2. 使用视觉大模型（如 GPT-4o）进行解析，得到 markdown 文件。



## 样例

有关 PDF，请参阅 [examples/attention_is_all_you_need/output.md](examples/attention_is_all_you_need/output.md) [examples/attention_is_all_you_need.pdf](examples/attention_is_all_you_need.pdf)。



## 安装

```bash
pip install gptpdf
```



## 使用

```python
from gptpdf import parse_pdf
api_key = 'Your OpenAI API Key'
content, image_paths = parse_pdf(pdf_path, api_key=api_key)
print(content)
```

更多内容请见 [test/test.py](test/test.py)



## API

**parse_pdf**(pdf_path, output_dir='./', api_key=None, base_url=None, model='gpt-4o', verbose=False)

将 pdf 文件解析为 markdown 文件，并返回 markdown 内容和所有图片路径列表。

- **pdf_path**：pdf 文件路径

- **output_dir**：输出目录。存储所有图片和 markdown 文件

- **api_key**：OpenAI API 密钥（可选）。如果未提供，则使用 OPENAI_API_KEY 环境变量。

- **base_url**：OpenAI 基本 URL。（可选）。如果未提供，则使用 OPENAI_BASE_URL 环境变量。

- **model**：OpenAI Vison LLM 模型，默认为“gpt-4o”。您也可以使用 qwen-vl-max

- **verbose**：详细模式

- **gpt_worker**: gpt解析工作线程数，默认为1. 如果您的机器性能较好，可以适当调高，以提高解析速度。



## 加入我们👏🏻

使用微信扫描下方二维码，加入微信群聊，或参与贡献。

<p align="center">
<img src="./docs/wechat.jpg" alt="wechat" width=400/>
</p>