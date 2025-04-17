import os
import dotenv
dotenv.load_dotenv()
api_key = os.getenv('api_key')
base_url = os.getenv('base_url')

pdf_path = '../examples/attention_is_all_you_need.pdf'
output_dir = '../examples/attention_is_all_you_need/'

pdf_path = '../examples/rh.pdf'
output_dir = '../examples/rh/'

# 清空output_dir
# import shutil
# shutil.rmtree(output_dir, ignore_errors=True)

def test_parse_pdf():
    from gptpdf import parse_pdf
    content, image_paths = parse_pdf(
        pdf_path, 
        output_dir=output_dir, 
        api_key=api_key, 
        base_url=base_url, 
        model='gpt-4o', 
        gpt_worker=6
        )
    print(content)
    print(image_paths)


if __name__ == '__main__':
    test_parse_pdf()