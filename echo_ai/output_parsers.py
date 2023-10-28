import re


class MyParser:
    """
    输出解析器
    解析文本并得到 在start_delimiter内容end_delimiter,之间的内容。
    """

    def parse(self, text: str, start_delimiter: str = '###', end_delimiter: str = '###'):
        pattern = re.escape(start_delimiter) + r'([\s\S]*?)' + re.escape(end_delimiter)
        matches = re.findall(pattern, text)
        content_list = [match.strip() for match in matches]
        return content_list
