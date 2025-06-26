try:
    HAS_DIFY = True
    from configs import dify_config
except:
    HAS_DIFY = False

# 支持的最低版本号
MIN_SUPPORTED_VERSION = "0.8.3"
# 支持的最高版本号
MAX_SUPPORTED_VERSION = "1.3.0"


def _compare_versions(version1, version2):
    """
    比较两个版本号
    Args:
        version1: 第一个版本号
        version2: 第二个版本号
    Returns:
        int: 如果version1 > version2返回1，如果version1 < version2返回-1，如果相等返回0
    """
    v1_parts = list(map(int, version1.split('.')))
    v2_parts = list(map(int, version2.split('.')))

    for v1, v2 in zip(v1_parts, v2_parts):
        if v1 > v2:
            return 1
        elif v1 < v2:
            return -1
    return 0


def is_version_supported():
    """
    检查当前版本是否支持
    Returns:
        bool: 如果版本支持返回True，否则返回False
    """
    if not HAS_DIFY:
        return False

    try:
        current_version = dify_config.CURRENT_VERSION
        # 检查是否在支持的版本范围内
        min_check = _compare_versions(current_version, MIN_SUPPORTED_VERSION)
        max_check = _compare_versions(current_version, MAX_SUPPORTED_VERSION)

        # 当前版本必须大于等于最小版本且小于等于最大版本
        return min_check >= 0 and max_check <= 0
    except:
        return False



