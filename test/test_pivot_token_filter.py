"""Dependency-free regression tests for the Python pivot-token filter.

Run from the repository root:
    python -m unittest discover -s test -p 'test_pivot_token_filter.py' -v

No torch/model/API imports, credentials, or generation are needed.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import logging
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pivot_token_filter_under_test', ROOT / 'src/goals/pivot_token_filter.py')
FILTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FILTER)
DIFF = FILTER.tokens_differ_meaningfully


def selected_functions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.Module(body=[future, *nodes], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class TokenFilterTests(unittest.TestCase):
    def test_attributes_and_fragments(self):
        cases = [
            ('s == s', '.lower', '[::-'),
            ('s == s', '.lower', '.upper'),
            ('s == s.', 'lower', 'upper'),
            ('s == s.lo', 'wer', 'ad'),
            ('n.', 'bit_count', 'bit_length'),
            ('x.', 'lower', 'upper'),
            ('x.', '속성', '다른속성'),
        ]
        for prefix, a, b in cases:
            with self.subTest(prefix=prefix, a=a, b=b):
                self.assertTrue(DIFF(a, b, prefix=prefix, variable_names=('s','n','x','lower','upper')))

    def test_literals_are_not_variable_names(self):
        cases = [
            ('s[0] == ', "'a'", "'b'"),
            ("s[0] == '", 'a', 'b'),
            ("s == '", ' ', ''),
            ('s == ', "' '", "''"),
            ('s == ', "'s'", "'t'"),
            ('s == """', 'a', 'b'),
            ("s == '''", 'a', 'b'),
            ('s == ', "'x y'", "'xy'"),
            ("s == r'", 'a', 'b'),
        ]
        for prefix, a, b in cases:
            with self.subTest(prefix=prefix, a=a, b=b):
                self.assertTrue(DIFF(a, b, prefix=prefix, variable_names=('a','b','s','t','x','y')))

    def test_known_input_aliases_only(self):
        for prefix, a, b, aliases in [
            ('', '(s', '(t', ('s','t')),
            ('', '[items', '[values', ('items','values')),
            ('', '(len', '(data', ('len','data')),
            ('(', '이름', '다른이름', ('이름','다른이름')),
            ('va', 'lue', 'l', ('value','val')),
        ]:
            with self.subTest(prefix=prefix, a=a, b=b):
                self.assertFalse(DIFF(a, b, prefix=prefix, variable_names=aliases))
        self.assertTrue(DIFF('(foo', '(bar'))  # not assumed to be variables
        self.assertTrue(DIFF('helper_a(', 'helper_b('))

    def test_formatting_outside_literals(self):
        cases = [
            ('n', ' ==', '=='),
            ('n ', '==', ' =='),
            ('', '(x', '( x'),
            ('x', ' # first', ' # second'),
            ('s == ', "'a'", " 'a'"),
            ('(', '\nx', ' x'),
            ('x', ' ', '\t'),
        ]
        for prefix, a, b in cases:
            with self.subTest(prefix=prefix, a=a, b=b):
                self.assertFalse(DIFF(a, b, prefix=prefix, variable_names=('x','n','s')))
        # Do not strip a model token before joining: this changes lexical boundaries.
        self.assertTrue(DIFF(' in', 'in', prefix='s', variable_names=('s',)))
        self.assertFalse(DIFF('==', ' ==', prefix='s',
                         dest_code='def f(s, x): return s == x',
                         targ_code='def f(x, s): return s == x'))

    def test_operators_keywords_and_values_survive(self):
        for a,b in [('<','>'),('2','3'),('True','False'),('not','len('),('min','max'),(')',' ]'),('[-','[')]:
            with self.subTest(a=a,b=b):
                self.assertTrue(DIFF(a,b,variable_names=('True','False')))
        self.assertFalse(DIFF('same','same'))

    def test_reference_binding_roles(self):
        left = 'def f(s):\n    return all(x > 0 for x in s)'
        right = 'def f(s):\n    return all(y > 0 for y in s)'
        kw = dict(prefix='all(', variable_names=('s',), dest_code=left, targ_code=right)
        self.assertFalse(DIFF('x','y',**kw))
        self.assertTrue(DIFF('s','y',**kw))  # input argument and local binder differ
        two = 'def f(s):\n    return all(x < y for x, y in zip(s, s[1:]))'
        self.assertTrue(DIFF('x','y',prefix='all(',dest_code=two,targ_code=two))
        renamed = 'def f(s):\n    return all(u < v for u, v in zip(s, s[1:]))'
        self.assertFalse(DIFF('x','u',prefix='all(',dest_code=two,targ_code=renamed))
        self.assertTrue(DIFF('x','unknown',prefix='all(',dest_code=left,targ_code=right))

    def test_incomplete_and_ambiguous_references_are_conservative(self):
        self.assertTrue(DIFF('foo','bar',prefix='(',dest_code='def f(',targ_code='def f('))
        nested1 = 'def f(s):\n    return list(map(lambda x: x, s))'
        nested2 = 'def f(s):\n    return list(map(lambda y: y, s))'
        self.assertTrue(DIFF('x','y',prefix='(',dest_code=nested1,targ_code=nested2))
        a = 'def f(s):\n    s = str(s)\n    return s'
        b = 'def f(t):\n    t = str(t)\n    return t'
        self.assertTrue(DIFF('s','t',dest_code=a,targ_code=b))


class WiringTests(unittest.TestCase):
    def test_every_semantic_filter_call_supplies_prefix_and_known_names(self):
        counts = {}
        for rel in ('src/goals/head_loc_oracle.py','src/diff/nrgs_prefix.py'):
            calls = [n for n in ast.walk(ast.parse((ROOT/rel).read_text()))
                     if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                     and n.func.id in ('_tokens_differ_meaningfully','tokens_differ_meaningfully')]
            counts[rel] = len(calls)
            for call in calls:
                keys={k.arg for k in call.keywords}
                self.assertIn('prefix',keys,(rel,call.lineno))
                self.assertIn('variable_names',keys,(rel,call.lineno))
        self.assertEqual(counts,{'src/goals/head_loc_oracle.py':3,'src/diff/nrgs_prefix.py':1})

    def test_completion_gate_reuses_two_codes_without_extra_calls(self):
        calls=[]
        def fake(mt,sample,lcp,docstring,max_tokens,codes):
            calls.append(sample)
            if codes is not None:codes.append(f'code:{sample}')
            return True
        ns=selected_functions(ROOT/'src/goals/head_loc_oracle.py',{'_completion_gate'}, {'_greedy_complete_and_eval':fake})
        codes=[]
        self.assertTrue(ns['_completion_gate'](None,'dst','targ','prefix',completion_codes=codes))
        self.assertEqual(calls,['dst','targ'])
        self.assertEqual(codes,['code:dst','code:targ'])

    def test_candidate_validator_uses_contextual_filter(self):
        dest=types.SimpleNamespace(variable_name='s', label='dst')
        targ=types.SimpleNamespace(variable_name='s', label='targ')
        codes=['def f(s): return s == s.lower()', 'def f(s): return s == s.upper()']
        tokens={'dst':'.lower','targ':'.upper'}
        def completion_gate(*args,completion_codes=None,**kwargs):
            completion_codes.extend(codes)
            return True
        ns=selected_functions(ROOT/'src/goals/head_loc_oracle.py',{'_validate_candidate'}, {
            '_completion_gate':completion_gate,
            'build_python_prompt':lambda mt,sample,**kwargs:sample.label,
            'get_next_token_probs':lambda mt,prompt:tokens[prompt],
            'get_top_token':lambda mt,probs:probs,
            '_tokens_differ_meaningfully':DIFF,
            'PivotInstance':types.SimpleNamespace,
            'logger':logging.getLogger('test'),
        })
        found=ns['_validate_candidate'](None,dest,targ,'s == s')
        self.assertEqual(found.dest_top_token,'.lower')
        self.assertEqual(found.target_top_token,'.upper')
        tokens.update(dst='==',targ=' ==')
        self.assertIsNone(ns['_validate_candidate'](None,dest,targ,'s'))

    def test_revalidation_clears_zero_survivors_only_for_selected_pairs(self):
        p1=types.SimpleNamespace(source_predicate='a',destination_predicate='b',pivot_instances=['stale'])
        p2=types.SimpleNamespace(source_predicate='c',destination_predicate='d',pivot_instances=['untouched'])
        saved=[]
        ds=types.SimpleNamespace(pairs=[p1,p2],get_sample=lambda name:object(),save_pairs=lambda name:saved.append(name))
        ns=selected_functions(ROOT/'src/goals/head_loc_oracle.py',{'get_oracle_pivot_instances'}, {'logger':logging.getLogger('test')})
        result=ns['get_oracle_pivot_instances'](types.SimpleNamespace(name='test/model'),ds,pairs=[p1],
                  include_case3=False,include_case3b=False,include_augmentation=False,save=True)
        self.assertEqual(result,[])
        self.assertEqual(p1.pivot_instances,[])
        self.assertEqual(p2.pivot_instances,['untouched'])
        self.assertEqual(saved,['model'])

    def test_old_nrgs_gate_cache_is_invalidated_and_preserved(self):
        tree=ast.parse((ROOT/'scripts/diff/mine_nrgs_prefix.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        stmts=[];collect=False
        for stmt in main.body:
            isassign=isinstance(stmt,ast.Assign)
            targets={t.id for t in stmt.targets if isinstance(t,ast.Name)} if isassign else set()
            if 'previous' in targets:collect=True
            if collect:stmts.append(stmt)
            if collect and 'done' in targets:break
        self.assertTrue(stmts)
        code=compile(ast.Module(body=stmts,type_ignores=[]),'cache_resume','exec')
        with tempfile.TemporaryDirectory() as directory:
            gate=Path(directory)/'gate.json'
            rows=[{'key':'old','passed':True},{'key':'new','token_filter_version':FILTER.TOKEN_FILTER_VERSION}]
            gate.write_text(json.dumps(rows))
            ns=dict(gate_fp=gate,json=json,logger=logging.getLogger('test'),TOKEN_FILTER_VERSION=FILTER.TOKEN_FILTER_VERSION)
            exec(code,ns)
            self.assertEqual(set(ns['done']),{'new'})
            backup=gate.with_name(f'gate.before-{FILTER.TOKEN_FILTER_VERSION}.json')
            self.assertEqual(json.loads(backup.read_text()),rows)


if __name__ == '__main__':
    unittest.main()
