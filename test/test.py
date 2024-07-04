import os

# laod environment variables from .env file
import dotenv
dotenv.load_dotenv()

pdf_path = '../examples/attention_is_all_you_need.pdf'
output_dir = '../examples/attention_is_all_you_need/'   

# 清空output_dir
import shutil
shutil.rmtree(output_dir, ignore_errors=True)

def test_use_api_key():
    from gptpdf import parse_pdf
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_API_BASE')
    # Manually provide OPENAI_API_KEY and OPEN_API_BASE
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, api_key=api_key, base_url=base_url, model='gpt-4o', gpt_worker=6)
    print(content)
    print(image_paths)
    # also output_dir/output.md is generated


def test_use_env():
    from gptpdf import parse_pdf
    
    # Use OPENAI_API_KEY and OPENAI_API_BASE from environment variables
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, model='gpt-4o', verbose=True)
    print(content)
    print(image_paths)
    # also output_dir/output.md is generated

def test_azure():
    from gptpdf import parse_pdf
    api_key = '8ef0b4df45e444079cd5a4xxxxx' # Azure API Key
    base_url = 'https://xxx.openai.azure.com/' # Azure API Base URL
    model = 'azure_xxxx' # azure_ with deploy ID name (not open ai model name), e.g. azure_cpgpt4
    # Use OPENAI_API_KEY and OPENAI_API_BASE from environment variables
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, api_key=api_key, base_url=base_url, model=model, verbose=True)
    print(content)
    print(image_paths)
    



if __name__ == '__main__':
    test_use_api_key()
    # test_use_env()
    # test_azure()