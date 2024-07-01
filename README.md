# gptpdf

<p align="center">
<a href="README_CN.md"><img src="https://img.shields.io/badge/文档-中文版-blue.svg" alt="CN doc"></a>
<a href="README.md"><img src="https://img.shields.io/badge/document-English-blue.svg" alt="EN doc"></a>
</p>

Using VLLM (like GPT-4o) to parse PDF into markdown.

Our approach is very simple (only 293 lines of code), but can almost perfectly parse typography, math formulas, tables, pictures, charts, etc.

Average cost per page: $0.013

This package use [GeneralAgent](https://github.com/CosmosShadow/GeneralAgent) lib to interact with OpenAI API.



## Process steps

1. Use the PyMuPDF library to parse the PDF to find all non-text areas and mark them, for example:

![](docs/demo.jpg)

2. Use a large visual model (such as GPT-4o) to parse and get a markdown file.



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

- **model**: OpenAI Vision Large Model, default is 'gpt-4o'. You also can use [qwen-vl-max](https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start), [GLM-4V](https://open.bigmodel.cn/dev/api#glm-4v) by change the `OPENAI_BASE_URL` or specify `base_url`.

- **verbose**: verbose mode

- **gpt_worker**: gpt parse worker number. default is 1. If your machine performance is good, you can increase it appropriately to improve parsing speed.
