# gptpdf
Using GPT to parse PDF

## Introduction

This package uses OpenAI's GPT-4o to parse PDFs to Markdowns.

It perfectly parse text, image, math equations, charts, and tables.

It almost cost $0.013 per page.

TODO: add parse work flow


## DEMO

See [examples/attention_is_all_you_need/output.md](examples/attention_is_all_you_need/output.md) for PDF [examples/attention_is_all_you_need.pdf](examples/attention_is_all_you_need.pdf).

## Installation

```bash
pip install gptpdf
```

## Usage

```python
from gptpdf import parse_pdf
pdf_path = '../examples/attention_is_all_you_need.pdf'
output_dir = '../examples/attention_is_all_you_need/'
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_API_BASE')
# Manually provide OPENAI_API_KEY and OPEN_API_BASE
content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, api_key=api_key, base_url=base_url, model='gpt-4o')
print(content)
print(image_paths)
# also output_dir/output.md is generated
```

```python
from gptpdf import parse_pdf
pdf_path = '../examples/attention_is_all_you_need.pdf'
output_dir = '../examples/attention_is_all_you_need/'
# Use OPENAI_API_KEY and OPENAI_API_BASE from environment variables
content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, model='gpt-4o', verbose=True)
print(content)
print(image_paths)
# also output_dir/output.md is generated
```

## API

```python
def parse_pdf(pdf_path, output_dir='./', api_key=None, base_url=None, model='gpt-4o', verbose=False):
    """
    parse pdf file to markdown file
    :param pdf_path: pdf file path
    :param output_dir: output directory. store all images and markdown file
    :param api_key: OpenAI API Key (optional). If not provided, Use OPENAI_API_KEY environment variable.
    :param base_url: OpenAI Base URL. (optional). If not provided, Use OPENAI_BASE_URL environment variable.
    :param model: OpenAI Vison LLM Model, default is 'gpt-4o'. You also can use qwen-vl-max
    :param verbose: verbose mode
    :return: markdown content with ![](path/to/image.png) and all rect image (image, table, chart, ...) paths.
    """

    """
    解析PDF文件到markdown文件
    :param pdf_path: pdf文件路径
    :param output_dir: 输出目录。存储所有的图片和markdown文件
    :param api_key: OpenAI API Key（可选）。如果未提供，则使用OPENAI_API_KEY环境变量。
    :param base_url: OpenAI Base URL。 （可选）。如果未提供，则使用OPENAI_BASE_URL环境变量。
    :param model: OpenAI Vison LLM Model，默认为'gpt-4o'。您还可以使用qwen-vl-max
    :param verbose: 详细模式，默认为False
    :return: (content, all_rect_images), markdown内容，带有![](path/to/image.png) 和 所有矩形图像（图像、表格、图表等）路径列表。
    """
```