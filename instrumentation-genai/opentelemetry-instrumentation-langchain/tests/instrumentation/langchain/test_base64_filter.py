import pytest
from unittest.mock import patch
from opentelemetry.instrumentation.langchain.internal._utils import _filter_base64_images, _is_base64_image, recursive_size


class TestBase64Filter:
    """测试base64图片过滤功能"""
    
    def test_is_base64_image(self):
        """测试base64图片检测功能"""
        base64_image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABA" + "a" * 12140}}
        normal_image = {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
        
        assert _is_base64_image(base64_image) is True
        assert _is_base64_image(normal_image) is False
        assert _is_base64_image({"text": "Hello world"}) is False
        assert _is_base64_image("not a dict") is False
        assert _is_base64_image(None) is False
    
    def test_recursive_size(self):
        """测试递归大小计算功能"""
        assert recursive_size("hello") > 0
        assert recursive_size([1, 2, 3]) > 0
        assert recursive_size({"a": 1, "b": 2}) > 0
        assert recursive_size({"level1": {"level2": {"data": [1, 2, 3, 4, 5]}}}) > 0
        assert recursive_size({"key": "x" * 20000}) > 10240
    
    def test_recursive_size_early_return(self):
        """测试recursive_size函数中的早退分支"""
        # 创建一个大的字典以触发早退
        large_dict = {f"key{i}": f"value{i}" * 100 for i in range(100)}
        size = recursive_size(large_dict, max_size=100)
        assert size > 100
        
        # 创建一个大的列表以触发早退
        large_list = [f"item{i}" * 100 for i in range(100)]
        size = recursive_size(large_list, max_size=100)
        assert size > 100
    
    @pytest.mark.parametrize("test_input,expected_filtered", [

        ({'prompts': ["Hello, this is a normal text without base64 images"]}, {'prompts': ["Hello, this is a normal text without base64 images"]}),
        # 包含小base64数据的情况
        ({
            'messages': [{'content': [{'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQ'}}], 'role': 'user'}], 
            'model': 'qwen-vl-max', 'stream': False
        }, {
            'messages': [{'content': [{'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQ'}}], 'role': 'user'}], 
            'model': 'qwen-vl-max', 'stream': False
        }),
        # 列表中的base64图片过滤
        ([
            {'type': 'text', 'text': '简述这个图片'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQSkZJR' + 'a' * 12140}}
        ], [
            {'type': 'text', 'text': '简述这个图片'},
            {'type': 'image_url', 'image_url': {'url': 'BASE64_IMAGE_DATA_FILTERED'}}
        ]),
        # 字符串中的base64图片过滤
        ({
            'prompts': ["Human: [{'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQ+"+"a"*12140+"'}}]"]
        }, {
            'prompts': ["Human: [{'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'BASE64_IMAGE_DATA_FILTERED'}}]"]
        }),
        # 边界情况
        (None, None), ([], []), ({}, {}), ("string", "string"), (123, 123),
        # 混合数据
        ({
            'normal_text': 'Hello',
            'base64_image': {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/SkZJRgABA'+ 'a' * 12140}},
            'normal_dict': {'key': 'value'}
        }, {
            'normal_text': 'Hello',
            'base64_image': {'type': 'image_url', 'image_url': {'url': 'BASE64_IMAGE_DATA_FILTERED'}},
            'normal_dict': {'key': 'value'}
        }),
        # 多个base64图片
        ({
            'content': [
                {'type': 'text', 'text': '第一张图片'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQSkZJRA' + 'a' * 12140}},
                {'type': 'text', 'text': '第二张图片'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgoA' + 'a' * 12140}}
            ]
        }, {
            'content': [
                {'type': 'text', 'text': '第一张图片'},
                {'type': 'image_url', 'image_url': {'url': 'BASE64_IMAGE_DATA_FILTERED'}},
                {'type': 'text', 'text': '第二张图片'},
                {'type': 'image_url', 'image_url': {'url': 'BASE64_IMAGE_DATA_FILTERED'}}
            ]
        })
    ])
    def test_filter_base64_images(self, test_input, expected_filtered):
        """测试base64图片过滤功能"""
        result = _filter_base64_images(test_input)
        assert result == expected_filtered
    
    def test_filter_base64_images_large_data(self):
        """测试大数据量进行过滤"""
        large_data = {
            'messages': [{'content': [{'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQ'}}], 'role': 'user'}], 
            'model': 'qwen-vl-max', 'stream': False
        }
        large_data['messages'][0]['content'][1]['image_url']['url'] = 'data:image/jpeg;base64,/9j/4AAQSk' + 'a' * 12140
        filtered = _filter_base64_images(large_data)
        assert filtered != large_data
        assert filtered['messages'][0]['content'][1]['image_url']['url'] == 'BASE64_IMAGE_DATA_FILTERED'
        
    def test_filter_base64_images_early_return(self):
        """测试_filter_base64_images函数中的早退分支"""
        # 测试没有找到'['或']'的情况
        miss_left = {'prompts': ["Human: {'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQ+"+"a"*12140+"'}}]"]}
        result = _filter_base64_images(miss_left)
        assert result == miss_left
        miss_right = {'prompts': ["Human: [{'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQ+"+"a"*12140+"'}}"]}
        result = _filter_base64_images(miss_right)
        assert result == miss_right


    def test_filter_base64_images_early_return_and_exception(self):
        """测试_filter_base64_images函数中的早退分支和异常处理"""
        
        # 创建一个大的对象以绕过早期大小检查，但格式无效以触发异常
        miss_right = {'prompts': ["Human: [{'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQ+"+"a"*20000+"'}}" + "a" * 10000]}
        result = _filter_base64_images(miss_right)
        # 应该返回原始对象，因为解析失败时保持原样
        assert result == miss_right

    # 新增mock测试用例
    def test_filter_base64_images_with_syntax_error_mock(self):
        """使用mock模拟ast.literal_eval抛出SyntaxError异常"""
        test_input = {'prompts': ["Human: [{'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQ+"+"a"*12140+"'}}]"]}
        
        with patch('opentelemetry.instrumentation.langchain.internal._utils.ast.literal_eval') as mock_literal_eval:
            mock_literal_eval.side_effect = SyntaxError("mocked syntax error")
            result = _filter_base64_images(test_input)
            # 应该返回原始对象，因为解析失败时保持原样
            assert result == test_input

    def test_filter_base64_images_early_return(self):
        """测试_filter_base64_images函数中的非期望类型直接返回"""

        miss = "Human: [{'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,/9j/4AAQ+" + "a" * 20000 + "'}}" + "a" * 10000
        result = _filter_base64_images(miss)
        # 应该返回原始对象，因为解析失败时保持原样
        assert result == miss