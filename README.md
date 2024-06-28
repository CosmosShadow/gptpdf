# gptpdf
Using GPT to parse PDF



## Installation

```bash
pip install gptpdf
```

## Usage

```python
from gptpdf import parse_pdf

pdf_path = "./examples/appattention_is_all_you_need.pdf"
api_key = "your-openai-api-key"
# base_url = "https://api.openai.com/v1"
model = 'gpt-4o'

content, image_paths = parse_pdf(pdf_path)
```