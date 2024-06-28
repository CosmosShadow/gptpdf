import os
from gptpdf import parse_pdf

path_load = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))
pdf_path = path_load('../examples/attention_is_all_you_need.pdf')
output_dir = path_load('../examples/attention_is_all_you_need/')


def test_use_api_key():
    api_key = 'your'
    base_url = 'your'
    content, image_paths = parse_pdf(pdf_path, output_dir, api_key, base_url)
    print(content)
    print(image_paths)


def test_use_env():
    import dotenv
    dotenv.load_dotenv(path_load('./.env'))
    content, image_paths = parse_pdf(pdf_path, output_dir)
    print(content)
    print(image_paths)


if __name__ == '__main__':
    # test_use_api_key()
    test_use_env()