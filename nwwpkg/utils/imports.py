# nwwpkg/utils/imports.py
import importlib

def load_callable(module_name: str, func_name: str):
    """
    지정한 모듈에서 특정 함수를 동적으로 불러오기
    예: load_callable("nwwpkg.ui.pages.scenarios", "page_scenarios")
    """
    try:
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"❌ 모듈 {module_name} 에서 {func_name} 함수를 불러올 수 없습니다: {e}"
        )
