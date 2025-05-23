
def handler_unicode(raw :str) -> str:
    row_text = bytes(raw, "utf-8").decode("unicode_escape")
    # 将解码的 Unicode 字符串编码为 UTF-8
    input_val = row_text.encode("utf-8")
    return input_val