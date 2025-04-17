import os
import re
import base64
from typing import List, Tuple, Optional, Dict
import fitz
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
import concurrent.futures
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# This Default Prompt Using Chinese and could be changed to other languages.

DEFAULT_PROMPT = """使用markdown语法，将图片中识别到的文字转换为markdown格式输出。你必须做到：
1. 输出和使用识别到的图片的相同的语言，例如，识别到英语的字段，输出的内容必须是英语。
2. 不要解释和输出无关的文字，直接输出图片中的内容。例如，严禁输出 "以下是我根据图片内容生成的markdown文本："这样的例子，而是应该直接输出markdown。
3. 内容不要包含在```markdown ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式、忽略掉长直线、忽略掉页码。
再次强调，不要解释和输出无关的文字，直接输出图片中的内容。
"""
DEFAULT_RECT_PROMPT = """图片中用红色框和名称(%s)标注出了一些区域。如果区域是表格或者图片，使用 ![]() 的形式插入到输出内容中，否则直接输出文字内容。
"""
DEFAULT_ROLE_PROMPT = """你是一个PDF文档解析器，使用markdown和latex语法输出图片的内容。
"""


def _is_near(rect1, rect2, distance = 20):
    """
    检查两个矩形是否靠近，如果它们之间的距离小于目标距离。
    @param rect1: 矩形1
    @param rect2: 矩形2
    @param distance: 目标距离
    @return: 是否靠近
    """
    return rect1.buffer(0.1).distance(rect2.buffer(0.1)) < distance


def _is_horizontal_near(rect1, rect2, distance = 100):
    """
    检查两个矩形是否水平靠近，如果其中一个矩形是水平线。
    @param rect1: 矩形1
    @param rect2: 矩形2
    @param distance: 目标距离
    @return: 是否水平靠近
    """
    result = False
    if abs(rect1.bounds[3] - rect1.bounds[1]) < 0.1 or abs(rect2.bounds[3] - rect2.bounds[1]) < 0.1:
        if abs(rect1.bounds[0] - rect2.bounds[0]) < 0.1 and abs(rect1.bounds[2] - rect2.bounds[2]) < 0.1:
            result = abs(rect1.bounds[3] - rect2.bounds[3]) < distance
    return result


def _union_rects(rect1, rect2):
    """
    合并两个矩形。
    @param rect1: 矩形1
    @param rect2: 矩形2
    @return: 合并后的矩形
    """
    return sg.box(*(rect1.union(rect2).bounds))


def _merge_rects(rect_list, distance = 20, horizontal_distance = None):
    """
    合并列表中的矩形，如果它们之间的距离小于目标距离。
    @param rect_list: 矩形列表
    @param distance: 目标距离
    @param horizontal_distance: 水平目标距离
    @return: 合并后的矩形列表
    """
    merged = True
    while merged:
        merged = False
        new_rect_list = []
        while rect_list:
            rect = rect_list.pop(0)
            for other_rect in rect_list:
                if _is_near(rect, other_rect, distance) or (
                        horizontal_distance and _is_horizontal_near(rect, other_rect, horizontal_distance)):
                    rect = _union_rects(rect, other_rect)
                    rect_list.remove(other_rect)
                    merged = True
            new_rect_list.append(rect)
        rect_list = new_rect_list
    return rect_list


def _adsorb_rects_to_rects(source_rects, target_rects, distance=10):
    """
    当距离小于目标距离时，将一组矩形吸附到另一组矩形。
    @param source_rects: 源矩形列表
    @param target_rects: 目标矩形列表
    @param distance: 目标距离
    @return: 吸附后的源矩形列表和目标矩形列表
    """
    new_source_rects = []
    for text_area_rect in source_rects:
        adsorbed = False
        for index, rect in enumerate(target_rects):
            if _is_near(text_area_rect, rect, distance):
                rect = _union_rects(text_area_rect, rect)
                target_rects[index] = rect
                adsorbed = True
                break
        if not adsorbed:
            new_source_rects.append(text_area_rect)
    return new_source_rects, target_rects


def _parse_rects(page):
    """
    解析页面中的绘图，并合并相邻的矩形。
    @param page: 页面
    @return: 矩形列表
    """

    # 提取画的内容
    drawings = page.get_drawings()

    # 忽略掉长度小于30的水平直线
    is_short_line = lambda x: abs(x['rect'][3] - x['rect'][1]) < 1 and abs(x['rect'][2] - x['rect'][0]) < 30
    drawings = [drawing for drawing in drawings if not is_short_line(drawing)]

    # 转换为shapely的矩形
    rect_list = [sg.box(*drawing['rect']) for drawing in drawings]

    # 提取图片区域
    images = page.get_image_info()
    image_rects = [sg.box(*image['bbox']) for image in images]

    # 合并drawings和images
    rect_list += image_rects

    merged_rects = _merge_rects(rect_list, distance=10, horizontal_distance=100)
    merged_rects = [rect for rect in merged_rects if explain_validity(rect) == 'Valid Geometry']

    # 将大文本区域和小文本区域分开处理: 大文本相小合并，小文本靠近合并
    is_large_content = lambda x: (len(x[4]) / max(1, len(x[4].split('\n')))) > 5
    small_text_area_rects = [sg.box(*x[:4]) for x in page.get_text('blocks') if not is_large_content(x)]
    large_text_area_rects = [sg.box(*x[:4]) for x in page.get_text('blocks') if is_large_content(x)]
    _, merged_rects = _adsorb_rects_to_rects(large_text_area_rects, merged_rects, distance=0.1) # 完全相交
    _, merged_rects = _adsorb_rects_to_rects(small_text_area_rects, merged_rects, distance=5) # 靠近

    # 再次自身合并
    merged_rects = _merge_rects(merged_rects, distance=10)

    # 过滤比较小的矩形
    merged_rects = [rect for rect in merged_rects if rect.bounds[2] - rect.bounds[0] > 20 and rect.bounds[3] - rect.bounds[1] > 20]

    return [rect.bounds for rect in merged_rects]


def _parse_pdf_to_images(pdf_path, output_dir = './'):
    """
    解析PDF文件到图片，并保存到输出目录。
    @param pdf_path: PDF文件路径
    @param output_dir: 输出目录
    @return: 图片信息列表(图片路径, 矩形图片路径列表)
    """
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    image_infos = []

    for page_index, page in enumerate(pdf_document):
        logging.info(f'parse page: {page_index}')
        rect_images = []
        rects = _parse_rects(page)
        for index, rect in enumerate(rects):
            fitz_rect = fitz.Rect(rect)
            # 保存页面为图片
            pix = page.get_pixmap(clip=fitz_rect, matrix=fitz.Matrix(4, 4))
            name = f'{page_index}_{index}.png'
            pix.save(os.path.join(output_dir, name))
            rect_images.append(name)
            # # 在页面上绘制红色矩形
            big_fitz_rect = fitz.Rect(fitz_rect.x0 - 1, fitz_rect.y0 - 1, fitz_rect.x1 + 1, fitz_rect.y1 + 1)
            # 空心矩形
            page.draw_rect(big_fitz_rect, color=(1, 0, 0), width=1)
            # 画矩形区域(实心)
            # page.draw_rect(big_fitz_rect, color=(1, 0, 0), fill=(1, 0, 0))
            # 在矩形内的左上角写上矩形的索引name，添加一些偏移量
            text_x = fitz_rect.x0 + 2
            text_y = fitz_rect.y0 + 10
            text_rect = fitz.Rect(text_x, text_y - 9, text_x + 80, text_y + 2)
            # 绘制白色背景矩形
            page.draw_rect(text_rect, color=(1, 1, 1), fill=(1, 1, 1))
            # 插入带有白色背景的文字
            page.insert_text((text_x, text_y), name, fontsize=10, color=(1, 0, 0))
        page_image_with_rects = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        page_image = os.path.join(output_dir, f'{page_index}.png')
        page_image_with_rects.save(page_image)
        image_infos.append((page_image, rect_images))

    pdf_document.close()
    return image_infos

def _remove_markdown_backticks(content: str) -> str:
    """
    删除markdown中的```字符串。
    """
    if '```markdown' in content:
        content = content.replace('```markdown\n', '')
        last_backticks_pos = content.rfind('```')
        if last_backticks_pos != -1:
            content = content[:last_backticks_pos] + content[last_backticks_pos + 3:]
    return content


def parse_pdf(
        pdf_path: str,
        output_dir: str = './',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = 'gpt-4o',
        gpt_worker: int = 1,
        prompt = DEFAULT_PROMPT,
        rect_prompt = DEFAULT_RECT_PROMPT,
        role_prompt = DEFAULT_ROLE_PROMPT,
) -> Tuple[str, List[str]]:
    """
    解析PDF文件到markdown文件。
    @param pdf_path: PDF文件路径
    @param output_dir: 输出目录
    @return: 解析后的markdown内容, 矩形图片路径列表
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_infos = _parse_pdf_to_images(pdf_path, output_dir=output_dir)
    
    # Process images with GPT
    def _process_page(index: int, image_info: Tuple[str, List[str]]) -> Tuple[int, str]:
        # 使用 OpenAI 客户端替代 Agent
        client = OpenAI(api_key=api_key, base_url=base_url)
        page_image, rect_images = image_info
        local_prompt = prompt
        if rect_images:
            local_prompt += rect_prompt + ', '.join(rect_images)
        
        # 打开图片文件
        with open(page_image, "rb") as image_file:
            # 调用 OpenAI API
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": role_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": local_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"}}
                        ]}
                    ]
                )

                # 检查 response.choices 是否为 None
                if not response.choices:
                    print(response)
                    return index, f"Error: Empty choices in API response for page {index+1}"
                    
                content = response.choices[0].message.content
                return index, content
            except Exception as e:
                # 捕获所有异常并返回错误信息
                return index, f"Error processing page {index+1}: {str(e)}"

    contents = [None] * len(image_infos)
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpt_worker) as executor:
        futures = [executor.submit(_process_page, index, image_info) for index, image_info in enumerate(image_infos)]
        for future in concurrent.futures.as_completed(futures):
            index, content = future.result()
            content = _remove_markdown_backticks(content)
            contents[index] = content

    # 保存解析后的markdown文件
    output_path = os.path.join(output_dir, 'output.md')
    content = '\n\n'.join(contents)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    #  删除中间过程的图片
    all_rect_images = []
    for page_image, rect_images in image_infos:
        if os.path.exists(page_image):
            os.remove(page_image)
        all_rect_images.extend(rect_images)

    return content, all_rect_images