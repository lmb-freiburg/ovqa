import pytest

from packg.testing.import_from_source import apply_visitor, ImportFromSourceChecker, recurse_modules

module_list = list(recurse_modules("ovqa", ignore_tests=True, packages_only=False))


@pytest.mark.parametrize("module", module_list)
def test_imports_from_source(module: str) -> None:
    print(f"Importing: {module}")
    apply_visitor(module=module, visitor=ImportFromSourceChecker(module))
