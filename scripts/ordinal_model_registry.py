"""Static discovery of TFB model exports, with no optional model imports.

This registry resolves names and adapters only. Runtime training capability is
checked separately; being discoverable is not a claim of ordinal-loss support.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'ts_benchmark.baselines.'
IDENTIFIER_PATH = re.compile(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z')


def _module_file(root, module):
    path = Path(root).joinpath(*module.split('.'))
    for candidate in (path.with_suffix('.py'), path/'__init__.py'):
        if candidate.is_file():
            return candidate
    return None


def _import_module(module, is_package, node):
    if not node.level:
        return node.module or ''
    package = module if is_package else module.rpartition('.')[0]
    parts = package.split('.')
    if node.level > len(parts):
        return ''
    base = '.'.join(parts[:len(parts)-node.level+1])
    return '.'.join(x for x in (base, node.module) if x)


def _symbols(root, module):
    path = _module_file(root, module)
    if path is None:
        return None, {}, None
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    symbols, public = {}, None
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols[node.name] = ('definition', node)
        elif isinstance(node, ast.ImportFrom):
            imported_module = _import_module(module, path.name == '__init__.py', node)
            for alias in node.names:
                if alias.name != '*':
                    symbols[alias.asname or alias.name] = ('import', imported_module+'.'+alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols[target.id] = ('definition', node)
                    if target.id == '__all__':
                        try:
                            value = ast.literal_eval(node.value)
                            if isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
                                public = list(value)
                        except (ValueError, TypeError, SyntaxError):
                            pass
    return path, symbols, public


def _definition(root, qualified, seen=None, allow_dynamic=False):
    seen = set() if seen is None else seen
    if qualified in seen:
        raise ValueError('Reexportacao circular de modelo: '+qualified)
    seen.add(qualified)
    module, _, symbol = qualified.rpartition('.')
    path, symbols, _ = _symbols(root, module)
    if path is not None and symbol not in symbols and allow_dynamic:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        if any(isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
               and n.value.func.id == 'globals' for n in ast.walk(tree)):
            return path, symbol, None
    if path is None or symbol not in symbols:
        raise ValueError('Modelo nao encontrado no checkout TFB: '+qualified)
    kind, value = symbols[symbol]
    if kind == 'import' and value.startswith(PREFIX):
        return _definition(root, value, seen, allow_dynamic=allow_dynamic)
    # Metadata dictionaries/factories are valid public TFB model information.
    return path, symbol, value if kind == 'definition' else None


def _is_public_model_name(name):
    # Without __all__, imported lowercase utilities are not advertised models.
    return (isinstance(name, str) and name and not name.startswith('_')
            and any(c.isupper() for c in name))


def _spec(root, qualified, allow_dynamic=False):
    source, class_name, definition = _definition(root, qualified, allow_dynamic=allow_dynamic)
    relative = qualified.removeprefix(PREFIX)
    family = relative.split('.')[0]
    # TSL exposes the shared four-input forecasting interface. Other packages
    # own their native wrapper; nn.Module alone does not imply that signature.
    adapter = 'transformer_adapter' if family == 'time_series_library' else None
    return dict(model_name=relative, adapter=adapter,
        deterministic='efficient' if class_name == 'TimesNet' else 'full',
        source_path=source.relative_to(root).as_posix(), class_name=class_name, family=family)


def discover_models(root=ROOT):
    """Map public qualified export names to metadata; dependency-free and dynamic."""
    root = Path(root).resolve()
    result = {}
    _, base_symbols, _ = _symbols(root, 'ts_benchmark.baselines')
    adapter_entry = base_symbols.get('ADAPTER')
    try:
        adapters = set(ast.literal_eval(adapter_entry[1].value)) if adapter_entry else set()
    except (ValueError, TypeError, AttributeError):
        adapters = set()
    for init in sorted((root/'ts_benchmark/baselines').glob('*/__init__.py')):
        module = '.'.join(init.parent.relative_to(root).parts)
        _, symbols, public = _symbols(root, module)
        explicit_exports = public is not None
        if public is None:
            public = [name for name, (kind, node) in symbols.items()
                      if kind == 'import' or isinstance(node, ast.ClassDef)]
        for name in public:
            if ((not explicit_exports and not _is_public_model_name(name))
                    or name in adapters or name not in symbols):
                continue
            qualified = module+'.'+name
            try:
                result[qualified] = _spec(root, qualified, allow_dynamic=True)
            except ValueError:
                # Broken exports are not silently routed to another model.
                continue
    return result


def _identity(spec):
    return spec['source_path'], spec['class_name']


def resolve_model(requested, root=ROOT, registry=None):
    """Resolve a unique short export or qualified local model information path.

    Returns (canonical_label, model_spec). Synonyms of the same implementation
    converge on one label so their training tasks are not duplicated.
    """
    root = Path(root).resolve()
    requested = 'Nonstationary_Transformer' if requested == 'NonStationary_Transformer' else requested
    if not isinstance(requested, str) or not IDENTIFIER_PATH.fullmatch(requested):
        raise ValueError('MODEL exige nome ou caminho Python qualificado: '+str(requested))
    registry = discover_models(root) if registry is None else registry
    bare = '.' not in requested
    if bare:
        candidates = [(path, spec) for path, spec in registry.items() if path.rsplit('.', 1)[-1] == requested]
        identities = {_identity(spec) for _, spec in candidates}
        if not candidates:
            raise ValueError('Modelo nao encontrado nos exports TFB: '+requested+
                             '; use o caminho Python completo se nao for exportado pelo pacote')
        if len(identities) > 1:
            options = ', '.join(path.removeprefix(PREFIX) for path, _ in candidates)
            raise ValueError('Nome de modelo ambiguo '+requested+'; use: '+options)
        qualified, spec = candidates[0]
    else:
        qualified = requested.removeprefix('global.')
        if not qualified.startswith(PREFIX):
            qualified = PREFIX+qualified
        spec = registry.get(qualified)
        if spec is None:
            # Fully qualified source names can name generated public factories.
            module, _, symbol = qualified.rpartition('.')
            source = _module_file(root, module)
            matches = [item for item in registry.values() if source is not None
                       and item['source_path'] == source.relative_to(root).as_posix()
                       and item['class_name'] == symbol]
            if matches:
                spec = matches[0]
            else:
                spec = _spec(root, qualified)
    aliases = [(path, item) for path, item in registry.items() if _identity(item) == _identity(spec)]
    if aliases:
        aliases.sort(key=lambda row: (row[0].rsplit('.', 1)[-1] != spec['class_name'], row[0]))
        qualified, spec = aliases[0]
        label = qualified.rsplit('.', 1)[-1]
        same_label = {_identity(item) for path, item in registry.items() if path.rsplit('.', 1)[-1] == label}
        if len(same_label) > 1:
            label = qualified.removeprefix(PREFIX)
    else:
        label = qualified.removeprefix(PREFIX)
    return label, dict(spec)
