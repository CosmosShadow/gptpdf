# gptpdf

<p align="center">
<a href="README_CN.md"><img src="https://img.shields.io/badge/文档-中文版-blue.svg" alt="CN doc"></a>
<a href="README.md"><img src="https://img.shields.io/badge/document-English-blue.svg" alt="EN doc"></a>
</p>

Using VLLM (like GPT-4o) to parse PDF into markdown.

Our approach is very simple (only 293 lines of code), but can almost perfectly parse typography, math formulas, tables, pictures, charts, etc.

Average price per page: $0.013

This package use [GeneralAgent](https://github.com/CosmosShadow/GeneralAgent) lib to interact with OpenAI API.



## Process steps

1. 使用 PyMuPDF 库，对 PDF 进行解析出所有非文本区域，并做好标记，比如:

![](docs/demo.jpg)

2. 使用视觉大模型（如 GPT-4o）进行解析，得到 markdown 文件。



## DEMO

See [examples/attention_is_all_you_need/output.md](examples/attention_is_all_you_need/output.md) for PDF [examples/attention_is_all_you_need.pdf](examples/attention_is_all_you_need.pdf).



## Installation

```bash
pip install gptpdf
```



## Usage

```python
from gptpdf import parse_pdf
api_key = 'Your OpenAI API Key'
content, image_paths = parse_pdf(pdf_path, api_key=api_key)
print(content)
```

See more in [test/test.py](test/test.py)



## API

**parse_pdf**(pdf_path, output_dir='./', api_key=None, base_url=None, model='gpt-4o', verbose=False)

parse pdf file to markdown file, and return markdown content and all image paths.

- **pdf_path**: pdf file path

- **output_dir**: output directory. store all images and markdown file

- **api_key**: OpenAI API Key (optional). If not provided, Use OPENAI_API_KEY environment variable.

- **base_url**: OpenAI Base URL. (optional). If not provided, Use OPENAI_BASE_URL environment variable.

- **model**: OpenAI Vison LLM Model, default is 'gpt-4o'. You also can use qwen-vl-max

- **verbose**: verbose mode