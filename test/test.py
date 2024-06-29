import os

# laod environment variables from .env file
import dotenv
dotenv.load_dotenv()

def test_use_api_key():
    from gptpdf import parse_pdf
    pdf_path = '../examples/attention_is_all_you_need.pdf'
    output_dir = '../examples/attention_is_all_you_need/'
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_API_BASE')
    # Manually provide OPENAI_API_KEY and OPEN_API_BASE
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, api_key=api_key, base_url=base_url, model='gpt-4o', gpt_worker=6)
    print(content)
    print(image_paths)
    # also output_dir/output.md is generated


def test_use_env():
    from gptpdf import parse_pdf
    pdf_path = '../examples/attention_is_all_you_need.pdf'
    output_dir = '../examples/attention_is_all_you_need/'
    # Use OPENAI_API_KEY and OPENAI_API_BASE from environment variables
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, model='gpt-4o', verbose=True)
    print(content)
    print(image_paths)
    # also output_dir/output.md is generated


if __name__ == '__main__':
    test_use_api_key()
    # test_use_env()