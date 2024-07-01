import fitz  # PyMuPDF
import shapely.geometry as sg
from shapely.validation import explain_validity
import concurrent.futures


def _is_near(rect1, rect2, distance=20):
    """
    check if two rectangles are near each other if the distance between them is less than the target
    """
    # 检测两个矩形的距离是否小于distance
    return rect1.buffer(1).distance(rect2.buffer(1)) < distance


def _is_horizontal_near(rect1, rect2, distance=100):
    """
    check if two rectangles are near horizontally if one of them is a horizontal line
    """
    result = False
    # rect1和rect2中有一个是水平线
    if abs(rect1.bounds[3] - rect1.bounds[1]) < 0.1 or abs(rect2.bounds[3] - rect2.bounds[1]) < 0.1:
        if abs(rect1.bounds[0] - rect2.bounds[0]) < 0.1 and abs(rect1.bounds[2] - rect2.bounds[2]) < 0.1:
            result = abs(rect1.bounds[3] - rect1.bounds[3]) < distance
    # print(rect1.bounds, rect2.bounds, result)
    return result


def _union_rects(rect1, rect2):
    """
    union two rectangles
    """
    # 合并两个矩形
    return sg.box(*(rect1.union(rect2).bounds))


def _merge_rects(rect_list, distance=20, horizontal_distance=None):
    """
    merge rectangles in the list if the distance between them is less than the target
    :param rect_list: list of rectangles
    :param distance: distance threshold
    :param horizontal_distance: horizontal distance threshold when one of the rectangles is a horizontal line
    """
    # 合并矩形列表
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
    adsorb a set of rectangles to another set of rectangles
    :param source_rects: source rectangle set
    :param target_rects: target rectangle set
    """
    """
    吸附一个集合到另外一个集合
    :param source_rects: 源矩形集合
    :param target_rects: 目标矩形集合
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
    return target_rects, new_source_rects


def _parse_drawings(page):
    """
    parse drawings in the page and merge adjacent rectangles
    """
    """
    解析页面中的绘图元素，合并相邻的矩形
    """
    drawings = page.get_drawings()

    # for drawing in drawings:
    #     print(drawing)

    rect_list = [drawing['rect'] for drawing in drawings]
    # 转成 shapely 的 矩形对象
    rect_list = [sg.box(rect[0], rect[1], rect[2], rect[3]) for rect in rect_list]

    merged_rects = _merge_rects(rect_list, distance=10, horizontal_distance=100)

    # 并删除无效的矩形
    merged_rects = [rect for rect in merged_rects if explain_validity(rect) == 'Valid Geometry']

    # 提取所有的字符串区域矩形框
    text_area_rects = []
    for text_area in page.get_text('dict')['blocks']:
        rect = text_area['bbox']
        text_area_rects.append(sg.box(rect[0], rect[1], rect[2], rect[3]))

    # 吸附文字到矩形区域
    merged_rects, text_area_rects = _adsorb_rects_to_rects(text_area_rects, merged_rects, distance=5)

    # 二次合并
    merged_rects = _merge_rects(merged_rects, distance=10)

    # 过滤掉高度 或者 宽度 不足 10 的矩形
    merged_rects = [rect for rect in merged_rects if
                    rect.bounds[2] - rect.bounds[0] > 10 and rect.bounds[3] - rect.bounds[1] > 10]

    # 将Polygon对象抽取bounds属性
    merged_rects = [rect.bounds for rect in merged_rects]

    return merged_rects


def _parse_images(page):
    """
    解析页面中的图片元素
    """
    images = page.get_image_info()
    return [image['bbox'] for image in images]


def _parse_tables(page):
    """
    解析页面中的表格元素
    """
    tables = page.find_tables(
        snap_tolerance=20,  # 调整容差以捕捉更多的表格线条
    )
    return [table.bbox for table in tables]


def _parse_rects(page):
    """
    parse rectangles in the page
    :param page: page object
    """
    """
    解析页面中的矩形元素
    """
    return _parse_images(page) + _parse_drawings(page)


def _parse_pdf_to_images(pdf_path, output_dir='./'):
    """
    parse pdf to images and save to output_dir
    :param pdf_path: pdf file path
    :param output_dir: output directory
    :return: image_infos [(page_image, rect_images)]
    """
    import os
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)

    image_infos = []
    for page_index, page in enumerate(pdf_document):
        print(f'parse page: {page_index}')
        # 保存页面为图片
        page_image = page.get_pixmap(matrix=fitz.Matrix(3, 3))

        rect_images = []
        # 解析页面中的矩形
        # if page_index != 5:
        #     continue
        rects = _parse_rects(page)
        # rects = _parse_tables(page)
        for index, rect in enumerate(rects):
            # print(page_index, index, rect)
            fitz_rect = fitz.Rect(rect)
            pix = page.get_pixmap(clip=fitz_rect, matrix=fitz.Matrix(4, 4))
            name = f'{page_index}_{index}.png'
            pix.save(os.path.join(output_dir, name))
            # 存储最简相对路径
            rect_images.append(name)

            # 在页面上绘制红色矩形(膨胀一个像素)
            # page.draw_rect(fitz_rect, color=(1, 0, 0), width=1)
            big_fitz_rect = fitz.Rect(fitz_rect.x0 - 1, fitz_rect.y0 - 1, fitz_rect.x1 + 1, fitz_rect.y1 + 1)
            page.draw_rect(big_fitz_rect, color=(1, 0, 0), width=1)

            # 在矩形内的左上角写上矩形的索引name，添加一些偏移量
            text_x = fitz_rect.x0 + 2  # 偏移量为2个单位
            text_y = fitz_rect.y0 + 10  # 偏移量为10个单位
            text_rect = fitz.Rect(text_x, text_y - 9, text_x + 80, text_y + 2)  # 创建一个白色背景的矩形

            # 绘制白色背景矩形
            page.draw_rect(text_rect, color=(1, 1, 1), fill=(1, 1, 1))

            # 插入带有白色背景的文字
            page.insert_text((text_x, text_y), name, fontsize=10, color=(1, 0, 0))

        # 重新生成带有矩形的页面图像
        page_image_with_rects = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        # 整页使用绝对路径
        page_image = os.path.join(output_dir, f'{page_index}.png')
        page_image_with_rects.save(page_image)

        image_infos.append((page_image, rect_images))

    # 关闭PDF文件
    pdf_document.close()
    return image_infos


def _gpt_parse_images(image_infos, output_dir='./', api_key=None, base_url=None, model='gpt-4o', verbose=False,
                      gpt_worker=1):
    """
    parse images to markdown content
    :param image_infos: [(page_image, rect_images)]
    :param output_dir: output directory
    :param api_key: OpenAI API Key
    :param base_url: OpenAI Base URL
    :param model: OpenAI Vison LLM Model
    :return: markdown content
    """
    import os
    from GeneralAgent import Agent
    # TODO： 提示词优化，适配更多模型。
    prompt = """
使用markdown语法，将要图片中识别到的文字转换为markdown格式是输出。你必须做到：
1. 输出和使用识别到的图片的相同的语言，例如，识别到英语的字段，输出的内容必须是英语。
2. 不要解释和输出无关的文字，直接输出图片中的内容。例如，严禁输出 “以下是我根据图片内容生成的markdown文本：”这样的例子，而是应该直接输出markdown。
3. 内容不要包含在```markdown ```中、段落公式使用 $$ $$ 的形式、行内公式使用 $ $ 的形式、忽略掉长直线、忽略掉页码。

再次强调，不要解释和输出无关的文字，直接输出图片中的内容。
"""
    # TODO： 提示词优化，适配更多模型。
    rect_prompt = """
图片中用红色框和名称(%s)标注出了一些区域。
如果区域是表格或者图片，使用 ![]() 的形式插入到输出内容中，否则直接输出文字内容。
"""

    role = '你是一个PDF文档解析器，使用markdown和latex语法输出图片的内容。'

    def _process_page(index, image_info):
        print(f'gpt parse page: {index}')
        agent = Agent(role=role, api_key=api_key, base_url=base_url, model=model, disable_python_run=True)
        page_image, rect_images = image_info
        local_prompt = prompt
        if rect_images:
            local_prompt += rect_prompt % ', '.join(rect_images)
        content = agent.run([local_prompt, {'image': page_image}], show_stream=verbose)
        return index, content

    contents = [None] * len(image_infos)
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpt_worker) as executor:
        futures = [executor.submit(_process_page, index, image_info) for index, image_info in enumerate(image_infos)]
        for future in concurrent.futures.as_completed(futures):
            index, content = future.result()
            contents[index] = content

    # 输出结果
    output_path = os.path.join(output_dir, 'output.md')
    with open(output_path, 'w',encoding='utf-8') as f:
        f.write('\n\n'.join(contents))

    return '\n\n'.join(contents)


def parse_pdf(pdf_path, output_dir='./', api_key=None, base_url=None, model='gpt-4o', verbose=False, gpt_worker=1):
    """
    parse pdf file to markdown file
    :param pdf_path: pdf file path
    :param output_dir: output directory. store all images and markdown file
    :param api_key: OpenAI API Key (optional). If not provided, Use OPENAI_API_KEY environment variable.
    :param base_url: OpenAI Base URL. (optional). If not provided, Use OPENAI_BASE_URL environment variable.
    :param model: OpenAI Vison LLM Model, default is 'gpt-4o'. You also can use qwen-vl-max
    :param verbose: verbose mode
    :param gpt_worker: gpt parse worker number
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
    :param gpt_worker: gpt解析工作线程数，默认为1
    :return: (content, all_rect_images), markdown内容，带有![](path/to/image.png) 和 所有矩形图像（图像、表格、图表等）路径列表。
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_infos = _parse_pdf_to_images(pdf_path, output_dir=output_dir)
    content = _gpt_parse_images(image_infos, output_dir=output_dir, api_key=api_key, base_url=base_url, model=model,
                                verbose=verbose, gpt_worker=gpt_worker)

    # 删除每页的图片 & 保留所有的矩形图片
    all_rect_images = []
    for page_image, rect_images in image_infos:
        if os.path.exists(page_image):
            os.remove(page_image)
        all_rect_images.extend(rect_images)

    return content, all_rect_images
