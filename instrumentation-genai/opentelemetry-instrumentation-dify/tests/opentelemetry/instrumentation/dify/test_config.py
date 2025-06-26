import unittest
from unittest.mock import patch, MagicMock
from opentelemetry.instrumentation.dify.config import is_version_supported, MIN_SUPPORTED_VERSION, MAX_SUPPORTED_VERSION

class TestConfig(unittest.TestCase):
    def setUp(self):
        # 在每个测试前重置 HAS_DIFY
        self.patcher = patch('opentelemetry.instrumentation.dify.config.HAS_DIFY', True)
        self.patcher.start()
        
    def tearDown(self):
        # 在每个测试后停止补丁
        self.patcher.stop()
        
    def test_version_supported_within_range(self):
        """测试版本在支持范围内的场景"""
        with patch('opentelemetry.instrumentation.dify.config.dify_config') as mock_config:
            # 测试中间版本
            mock_config.CURRENT_VERSION = "0.10.0"
            self.assertTrue(is_version_supported())
            
            # 测试最小版本
            mock_config.CURRENT_VERSION = MIN_SUPPORTED_VERSION
            self.assertTrue(is_version_supported())
            
            # 测试最大版本
            mock_config.CURRENT_VERSION = MAX_SUPPORTED_VERSION
            self.assertTrue(is_version_supported())
            
    def test_version_not_supported(self):
        """测试版本不在支持范围内的场景"""
        with patch('opentelemetry.instrumentation.dify.config.dify_config') as mock_config:
            # 测试低于最小版本
            mock_config.CURRENT_VERSION = "0.8.2"
            self.assertFalse(is_version_supported())

            # 测试高于最大版本
            mock_config.CURRENT_VERSION = "0.15.4"
            self.assertFalse(is_version_supported())

            # 测试高于最大版本
            mock_config.CURRENT_VERSION = "1.1.0"
            self.assertFalse(is_version_supported())
            
    def test_has_dify_false(self):
        """测试 HAS_DIFY 为 False 的场景"""
        with patch('opentelemetry.instrumentation.dify.config.HAS_DIFY', False):
            self.assertFalse(is_version_supported())
            
    def test_invalid_version_format(self):
        """测试版本号格式无效的场景"""
        with patch('opentelemetry.instrumentation.dify.config.dify_config') as mock_config:
            # 测试非数字版本号
            mock_config.CURRENT_VERSION = "invalid.version"
            self.assertFalse(is_version_supported())
            
            # 测试不完整的版本号
            mock_config.CURRENT_VERSION = "0.8"
            self.assertFalse(is_version_supported())
            
    def test_version_comparison_edge_cases(self):
        """测试版本比较的边缘情况"""
        with patch('opentelemetry.instrumentation.dify.config.dify_config') as mock_config:
            # 测试不同位数的版本号
            mock_config.CURRENT_VERSION = "0.8.3.0"
            self.assertTrue(is_version_supported())
            
            # 测试带前导零的版本号
            mock_config.CURRENT_VERSION = "0.08.3"
            self.assertTrue(is_version_supported())

if __name__ == '__main__':
    unittest.main() 