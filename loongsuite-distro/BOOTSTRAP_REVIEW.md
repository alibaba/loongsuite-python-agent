# bootstrap.py 流程梳理与优化建议

## 一、基本流程梳理

### 1. 安装流程 (`install_from_tar`)

```
main() 
  └─> install_from_tar()
      ├─> resolve_tar_path()          # 解析 tar 路径，可能需要下载
      ├─> extract_tar()               # 解压 tar 文件，获取所有 .whl 文件
      ├─> filter_packages()           # 过滤包（核心逻辑）
      │   ├─> get_package_name_from_whl()      # 从 whl 文件名提取包名
      │   ├─> _is_instrumentation_in_bootstrap_gen()  # 检查是否为 instrumentation
      │   ├─> check_python_version_compatibility()    # 检查 Python 版本兼容性
      │   ├─> check_dependency_compatibility()        # 检查依赖版本兼容性
      │   └─> get_target_libraries_from_bootstrap_gen() + _is_library_installed()  # 自动检测
      └─> install_packages()          # 使用 pip 安装
```

### 2. 卸载流程 (`uninstall_loongsuite_packages`)

```
main()
  └─> uninstall_loongsuite_packages()
      ├─> get_installed_loongsuite_packages()  # 获取已安装的包列表
      └─> uninstall_packages()                 # 使用 pip 卸载
```

### 3. 核心辅助函数

- **包名处理**：
  - `get_package_name_from_whl()`: 从 whl 文件名提取包名
  - `get_installed_package_version()`: 获取已安装包的版本（处理下划线/连字符变体）

- **元数据提取**：
  - `get_metadata_from_whl()`: 从 whl 文件提取 METADATA
  - `get_python_requirement_from_whl()`: 提取 Python 版本要求

- **兼容性检查**：
  - `check_python_version_compatibility()`: 检查 Python 版本
  - `check_dependency_compatibility()`: 检查依赖版本

- **bootstrap_gen 查询**：
  - `_is_instrumentation_in_bootstrap_gen()`: 检查是否为 instrumentation
  - `get_target_libraries_from_bootstrap_gen()`: 获取目标库列表

- **库检测**：
  - `_is_library_installed()`: 检查库是否已安装

## 二、发现的问题和优化建议

### 1. 🔴 包名规范化逻辑重复

**问题**：
- `get_installed_package_version()` 中三次尝试（原始名、下划线→连字符、连字符→下划线）
- `_is_library_installed()` 中也有类似逻辑
- 多个地方都有 `normalized_name = package_name.replace("_", "-")` 的重复

**建议**：
```python
def normalize_package_name(package_name: str) -> str:
    """统一规范化包名：将下划线转换为连字符"""
    return package_name.replace("_", "-")

def get_package_name_variants(package_name: str) -> List[str]:
    """获取包名的所有可能变体（用于查找）"""
    normalized = normalize_package_name(package_name)
    variants = [package_name]
    if normalized != package_name:
        variants.append(normalized)
    # 如果需要，也可以添加反向变体
    return variants
```

### 2. 🔴 从 requirement 字符串提取包名的逻辑重复

**问题**：
在 `_is_instrumentation_in_bootstrap_gen()` 和 `get_target_libraries_from_bootstrap_gen()` 中都有：
```python
default_pkg_name = (
    default_instr.split("==")[0]
    .split(">=")[0]
    .split("<=")[0]
    .split("~=")[0]
    .split("!=")[0]
    .strip()
)
```

**建议**：
```python
def extract_package_name_from_requirement(req_str: str) -> str:
    """从 requirement 字符串中提取包名"""
    try:
        return Requirement(req_str).name
    except Exception:
        # Fallback: 手动解析
        for op in ["==", ">=", "<=", "~=", "!=", ">", "<"]:
            if op in req_str:
                return req_str.split(op)[0].strip()
        return req_str.strip()
```

### 3. 🟡 get_installed_package_version 中的重复代码

**问题**：
三个几乎相同的 try-except 块，只是包名不同。

**建议**：
```python
def get_installed_package_version(package_name: str) -> Optional[str]:
    """获取已安装包的版本"""
    variants = get_package_name_variants(package_name)
    
    for variant in variants:
        version = _try_get_version(variant)
        if version:
            return version
    return None

def _try_get_version(package_name: str) -> Optional[str]:
    """尝试获取单个包名变体的版本"""
    cmd = [sys.executable, "-m", "pip", "show", package_name]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return None
```

### 4. 🟡 filter_packages 函数过长

**问题**：
`filter_packages()` 函数有 130+ 行，包含太多逻辑，可读性差。

**建议**：
拆分为多个小函数：
```python
def filter_packages(...):
    """主函数，协调各个过滤步骤"""
    base_packages = []
    instrumentation_packages = []
    
    for whl_file in whl_files:
        package_name = get_package_name_from_whl(whl_file)
        
        if _should_skip_package(package_name, whl_file, blacklist, whitelist, 
                                skip_version_check, auto_detect):
            continue
            
        if package_name in BASE_DEPENDENCIES:
            base_packages.append(whl_file)
        else:
            if _should_install_instrumentation(package_name, whl_file, auto_detect):
                instrumentation_packages.append(whl_file)
    
    return base_packages, instrumentation_packages

def _should_skip_package(...) -> bool:
    """检查是否应该跳过该包"""
    # 黑名单/白名单检查
    # Python 版本检查
    # 依赖版本检查
    pass

def _should_install_instrumentation(...) -> bool:
    """检查是否应该安装该 instrumentation"""
    # auto-detect 逻辑
    pass
```

### 5. 🟡 包名匹配逻辑重复

**问题**：
多处都有 `normalized_name == package_name or default_pkg_name == package_name` 这样的匹配。

**建议**：
```python
def package_names_match(name1: str, name2: str) -> bool:
    """检查两个包名是否匹配（考虑规范化）"""
    normalized1 = normalize_package_name(name1)
    normalized2 = normalize_package_name(name2)
    return (normalized1 == normalized2 or 
            name1 == name2 or 
            normalized1 == name2 or 
            name1 == normalized2)
```

### 6. 🟢 常量提取

**问题**：
`EXCLUDED_PACKAGES` 在函数内部定义，应该移到模块级别。

**建议**：
```python
# 在模块级别定义
UNINSTALL_EXCLUDED_PACKAGES = {
    "loongsuite-distro",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-instrumentation",
}
```

### 7. 🟢 错误处理改进

**问题**：
多处使用 `except Exception: pass`，可能隐藏重要错误。

**建议**：
更具体地捕获异常，至少记录警告：
```python
except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
    logger.debug(f"Failed to get version for {package_name}: {e}")
    return None
except Exception as e:
    logger.warning(f"Unexpected error getting version for {package_name}: {e}")
    return None
```

### 8. 🟢 使用 packaging 库解析 requirement

**问题**：
手动解析 requirement 字符串（split("==")[0]...）不够健壮。

**建议**：
统一使用 `packaging.requirements.Requirement` 解析（已经在用，但有些地方还在手动解析）。

### 9. 🟡 模块化建议

**建议**将代码拆分为多个模块**：

```
loongsuite/distro/
  ├── bootstrap.py          # 主入口和 CLI
  ├── package_utils.py      # 包名处理、版本获取等工具函数
  ├── metadata.py           # whl 元数据提取
  ├── compatibility.py     # 兼容性检查
  └── bootstrap_gen.py     # bootstrap_gen 查询（已存在）
```

## 三、优先级建议

### 高优先级（立即优化）
1. ✅ 提取包名规范化函数（减少重复，提高一致性）
2. ✅ 提取 requirement 解析函数（多处使用，容易出错）
3. ✅ 简化 `get_installed_package_version()`（消除重复代码）

### 中优先级（后续优化）
4. ⚠️ 拆分 `filter_packages()` 函数（提高可读性）
5. ⚠️ 提取包名匹配函数（统一匹配逻辑）
6. ⚠️ 改进错误处理（更好的调试体验）

### 低优先级（可选）
7. 💡 模块化拆分（如果文件继续增长）
8. 💡 使用更专业的 metadata 解析库（如果遇到解析问题）

## 四、总结

当前代码功能完整，但存在以下主要问题：
1. **代码重复**：包名规范化、requirement 解析等逻辑在多处重复
2. **函数过长**：`filter_packages()` 函数包含太多逻辑
3. **错误处理**：过于宽泛的异常捕获可能隐藏问题

建议优先解决代码重复问题，这将提高代码的可维护性和一致性。

