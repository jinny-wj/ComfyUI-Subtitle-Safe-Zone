import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_package():
    spec = importlib.util.spec_from_file_location(
        "safezone_contract_test", ROOT/"__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ContractTests(unittest.TestCase):
    def test_original_workflow_interface_unchanged(self):
        module = load_package()
        expected = json.loads((ROOT/"tests"/"interface_v0.1.0.json").read_text())
        self.assertIn(expected["node_id"], module.NODE_CLASS_MAPPINGS)
        cls = module.NODE_CLASS_MAPPINGS[expected["node_id"]]
        current = {"node_id": expected["node_id"], "inputs": cls.INPUT_TYPES(),
                   "return_types": cls.RETURN_TYPES, "return_names": cls.RETURN_NAMES,
                   "function": cls.FUNCTION}
        current = json.loads(json.dumps(current))
        self.assertEqual(current, expected)
        for group in ("required", "optional"):
            self.assertEqual(list(current["inputs"][group]), list(expected["inputs"][group]))

    def test_import_without_third_party_dependencies_and_failure_is_local(self):
        # Fresh interpreter with site-packages disabled. The finder also blocks
        # externally injected numpy/torch; registration must never request them.
        code = '''
import importlib.abc, importlib.util, sys
class BlockDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'torch'):
            raise ModuleNotFoundError('deliberately unavailable', name=fullname)
sys.meta_path.insert(0, BlockDependencies())
root = sys.argv[1]
spec = importlib.util.spec_from_file_location('safezone_test', root+'/__init__.py', submodule_search_locations=[root])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'numpy' not in sys.modules and 'torch' not in sys.modules
node = module.NODE_CLASS_MAPPINGS['SubtitleSafeZonePlanner']()
assert node.INPUT_TYPES()['required']['images'][0] == 'IMAGE'
try:
    node.plan(None)
except RuntimeError as exc:
    assert '不会安装' in str(exc)
else:
    raise AssertionError('Missing dependencies should give an actionable error')
print('Registration works without dependencies; execution fails locally.')
'''
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(ROOT)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inventory_works_without_third_party_dependencies(self):
        result = subprocess.run([sys.executable, "-I", "-S", "-B",
                                 str(ROOT/"tools"/"check_environment.py")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["compatibility_status"], "not_verified")
        self.assertFalse(report["packages"]["torch"]["module_found"])
        self.assertFalse(report["syntax_errors"])


if __name__ == "__main__":
    unittest.main()
