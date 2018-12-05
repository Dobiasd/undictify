"""
undictify - tests
"""

import json
import pickle
import unittest
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union, Tuple
from typing import TypeVar

from ._unpack import type_checked_call, type_checked_constructor

TypeT = TypeVar('TypeT')


class WithTwoMembers(NamedTuple):  # pylint: disable=too-few-public-methods
    """Some dummy class as a NamedTuple."""
    val: int
    msg: str


def create_with_two_members(val: int, msg: str) -> WithTwoMembers:
    """Some dummy function. Forwarding to class for value storage."""
    return WithTwoMembers(val, msg)


class TestArgsCalls(unittest.TestCase):  # pylint: disable=too-many-public-methods
    """Tests function calls with positional and keywords arguments."""

    def check_result(self, the_obj: WithTwoMembers) -> None:
        """Validate content of WithTwoMembers's members."""
        self.assertEqual(the_obj.val, 42)
        self.assertEqual(the_obj.msg, 'hello')

    def test_foo_function_positional(self) -> None:
        """Positional arguments only."""
        result: WithTwoMembers = type_checked_call()(create_with_two_members)(42, 'hello')
        self.check_result(result)

    def test_foo_function_keyword(self) -> None:
        """Keyword arguments only."""
        result: WithTwoMembers = type_checked_call()(create_with_two_members)(val=42, msg='hello')
        self.check_result(result)

    def test_foo_function_positional_and_keyword(self) -> None:  # pylint: disable=invalid-name
        """Positional and keyword arguments."""
        result: WithTwoMembers = type_checked_call()(create_with_two_members)(
            42, **{'msg': 'hello'})
        self.check_result(result)

    def test_foo_function_positional_and_keyword_duplicates(self) -> None:  # pylint: disable=invalid-name
        """Invalid (overlapping) combination of
        positional arguments and keyword arguments."""
        with self.assertRaises(TypeError):
            type_checked_call()(create_with_two_members)(42, 'hello', val=42)


class Foo:  # pylint: disable=too-few-public-methods
    """Some dummy class."""

    def __init__(self,  # pylint: disable=too-many-arguments,line-too-long
                 val: int, msg: str, flag: bool, opt: Optional[int],
                 frac: float) -> None:
        self.val: int = val
        self.msg: str = msg
        self.frac: float = frac
        self.flag: bool = flag
        self.opt: Optional[int] = opt


@type_checked_constructor()
class FooDecorated:  # pylint: disable=too-few-public-methods
    """Some dummy class."""

    def __init__(self,  # pylint: disable=too-many-arguments,line-too-long
                 val: int, msg: str, flag: bool, opt: Optional[int],
                 frac: float) -> None:
        self.val: int = val
        self.msg: str = msg
        self.frac: float = frac
        self.flag: bool = flag
        self.opt: Optional[int] = opt


class FooNamedTuple(NamedTuple):  # pylint: disable=too-few-public-methods
    """Some dummy class as a NamedTuple."""
    val: int
    msg: str
    frac: float
    flag: bool
    opt: Optional[int]


@type_checked_constructor()  # pylint: disable=too-few-public-methods
class FooNamedTupleDecorated(NamedTuple):
    """Some dummy class as a NamedTuple."""
    val: int
    msg: str
    frac: float
    flag: bool
    opt: Optional[int]


def foo_function(val: int, msg: str, frac: float, flag: bool,
                 opt: Optional[int]) -> Foo:
    """Some dummy function. Forwarding to class for value storage."""
    return Foo(val, msg, flag, opt, frac)


@type_checked_call()
def foo_function_type_checked_call(val: int, msg: str, frac: float, flag: bool,  # pylint: disable=invalid-name
                                   opt: Optional[int]) -> Foo:
    """Some dummy function. Forwarding to class for value storage."""
    return Foo(val, msg, flag, opt, frac)


@type_checked_call(convert=True)
def foo_function_type_checked_call_convert(val: int, msg: str, frac: float,  # pylint: disable=invalid-name
                                           flag: bool, opt: Optional[int]) -> Foo:
    """Some dummy function. Forwarding to class for value storage."""
    return Foo(val, msg, flag, opt, frac)


@type_checked_call(skip=True)
def foo_function_type_checked_call_skip(val: int, msg: str, frac: float,  # pylint: disable=invalid-name
                                        flag: bool, opt: Optional[int]) -> Foo:
    """Some dummy function. Forwarding to class for value storage."""
    return Foo(val, msg, flag, opt, frac)


@type_checked_call(skip=True, convert=True)
def foo_function_type_checked_call_skip_convert(val: int, msg: str, frac: float,  # pylint: disable=invalid-name
                                                flag: bool, opt: Optional[int]) -> Foo:
    """Some dummy function. Forwarding to class for value storage."""
    return Foo(val, msg, flag, opt, frac)


class TestUnpackingFoo(unittest.TestCase):  # pylint: disable=too-many-public-methods
    """Tests unpacking into ordinary class and into NamedTuple class."""

    def check_result(self, the_foo: Any, opt_val: Optional[int],
                     frac: float = 3.14,
                     msg: str = 'hello') -> None:
        """Validate content of Foo's members."""
        self.assertEqual(the_foo.val, 42)
        self.assertEqual(the_foo.msg, msg)
        self.assertEqual(the_foo.frac, frac)
        self.assertEqual(the_foo.flag, True)
        self.assertEqual(the_foo.opt, opt_val)

    def do_test_dict(self, func: Callable[..., TypeT],
                     decorated: bool) -> None:
        """Valid data dict."""
        data = {
            "val": 42, "msg": "hello", "frac": 3.14, "flag": True, "opt": 10}
        if not decorated:
            a_foo = type_checked_call()(func)(**data)
        else:
            a_foo = func(**data)
        self.check_result(a_foo, 10)

    def do_test_ok(self, func: Callable[..., TypeT],
                   decorated: bool) -> None:
        """Valid JSON string."""
        object_repr = '{"val": 42, "msg": "hello", "frac": 3.14, ' \
                      '"flag": true, "opt": 10}'
        if not decorated:
            a_foo = type_checked_call()(func)(**json.loads(object_repr))
        else:
            a_foo = func(**json.loads(object_repr))
        self.check_result(a_foo, 10)

    def do_test_opt_null(self, func: Callable[..., TypeT],
                         decorated: bool) -> None:
        """Valid JSON string null for the optional member."""
        object_repr = '{"val": 42, "msg": "hello", "frac": 3.14, ' \
                      '"flag": true, "opt": null}'
        if not decorated:
            a_foo = type_checked_call()(func)(**json.loads(object_repr))
        else:
            a_foo = func(**json.loads(object_repr))
        self.check_result(a_foo, None)

    def do_test_additional(self, func: Callable[..., TypeT],
                           decorated: bool) -> None:
        """Valid JSON string with an additional field."""
        object_repr = '{"val": 42, "msg": "hello", "frac": 3.14, ''' \
                      '"flag": true, "opt": 10, "ignore": 1}'
        if not decorated:
            a_foo = type_checked_call(skip=True)(func)(**json.loads(object_repr))
        else:
            a_foo = func(**json.loads(object_repr))
        self.check_result(a_foo, 10)

    def do_test_convert_ok(self, func: Callable[..., TypeT],
                           decorated: bool) -> None:
        """Valid JSON string."""
        object_repr = '{"val": "42", "msg": 5, "frac": 3, ' \
                      '"flag": true, "opt": 10.1}'
        if not decorated:
            a_foo = type_checked_call(convert=True)(func)(**json.loads(object_repr))
        else:
            a_foo = func(**json.loads(object_repr))
        self.check_result(a_foo, 10, 3.0, '5')

    def do_test_additional_and_convert(self, func: Callable[..., TypeT],
                                       decorated: bool) -> None:
        """Valid JSON string with an additional field and one to convert."""
        object_repr = '{"val": "42", "msg": "hello", "frac": 3.14, ''' \
                      '"flag": true, "opt": 10, "ignore": 1}'
        if not decorated:
            a_foo = type_checked_call(skip=True, convert=True)(func)(**json.loads(object_repr))
        else:
            a_foo = func(**json.loads(object_repr))
        self.check_result(a_foo, 10)

    @staticmethod
    def do_test_wrong_opt_type(func: Callable[..., TypeT],
                               decorated: bool) -> None:
        """Valid JSON string."""
        object_repr = '{"val": 42, "msg": "hello", "frac": 3.14, ' \
                      '"flag": true, "opt": "wrong"}'
        if not decorated:
            type_checked_call()(func)(**json.loads(object_repr))
        else:
            func(**json.loads(object_repr))

    @staticmethod
    def do_test_convert_error(func: Callable[..., TypeT],
                              decorated: bool) -> None:
        """Valid JSON string."""
        object_repr = '{"val": "twentyfour", "msg": "hello", "frac": 3.14, ' \
                      '"flag": true, "opt": 10}'
        if not decorated:
            type_checked_call(convert=True)(func)(**json.loads(object_repr))
        else:
            func(**json.loads(object_repr))

    @staticmethod
    def do_test_missing(func: Callable[..., TypeT],
                        decorated: bool) -> None:
        """Invalid JSON string: missing a field."""
        object_repr = '{"val": 42, "msg": "hello", "opt": 10, "flag": true}'
        if not decorated:
            type_checked_call()(func)(**json.loads(object_repr))
        else:
            func(**json.loads(object_repr))

    def do_test_opt_missing(self, func: Callable[..., TypeT],
                            decorated: bool) -> None:
        """Valid JSON string without providing value for optional member."""
        object_repr = '{"val": 42, "msg": "hello", "frac": 3.14, "flag": true}'
        if not decorated:
            a_foo = type_checked_call()(func)(**json.loads(object_repr))
        else:
            a_foo = func(**json.loads(object_repr))
        self.check_result(a_foo, None)

    @staticmethod
    def do_test_incorrect_type(func: Callable[..., TypeT],
                               decorated: bool) -> None:
        """Invalid JSON string: incorrect type of a field."""
        object_repr = '{"val": 42, "msg": "hello", "opt": 10, ' \
                      '"frac": "incorrect", "flag": true}'
        if not decorated:
            type_checked_call()(func)(**json.loads(object_repr))
        else:
            func(**json.loads(object_repr))

    def do_test_function_with_targets(self, func: Callable[..., TypeT],
                                      targets: List[Tuple[Any,
                                                          bool,
                                                          Any]]) -> None:
        """Run test function with provided target functions
        and expected exceptions."""
        for target, decorated, ex in targets:
            if ex:
                with self.assertRaises(ex):
                    func(target, decorated)
            else:
                func(target, decorated)

    def test_dict(self) -> None:
        """Valid data dict."""
        self.do_test_function_with_targets(self.do_test_dict, [
            (Foo, False, None),
            (FooDecorated, True, None),
            (FooNamedTuple, False, None),
            (FooNamedTupleDecorated, True, None),
            (foo_function, False, None),
            (foo_function_type_checked_call, True, None),
            (foo_function_type_checked_call_skip, True, None),
            (foo_function_type_checked_call_convert, True, None),
            (foo_function_type_checked_call_skip_convert, True, None),
        ])

    def test_ok(self) -> None:
        """Valid JSON string."""
        self.do_test_function_with_targets(self.do_test_ok, [
            (Foo, False, None),
            (FooDecorated, True, None),
            (FooNamedTuple, False, None),
            (FooNamedTupleDecorated, True, None),
            (foo_function, False, None),
            (foo_function_type_checked_call, True, None),
            (foo_function_type_checked_call_skip, True, None),
            (foo_function_type_checked_call_convert, True, None),
            (foo_function_type_checked_call_skip_convert, True, None),
        ])

    def test_opt_null(self) -> None:
        """Valid JSON string null for the optional member."""
        self.do_test_function_with_targets(self.do_test_opt_null, [
            (Foo, False, None),
            (FooDecorated, True, None),
            (FooNamedTuple, False, None),
            (FooNamedTupleDecorated, True, None),
            (foo_function, False, None),
            (foo_function_type_checked_call, True, None),
            (foo_function_type_checked_call_skip, True, None),
            (foo_function_type_checked_call_convert, True, None),
            (foo_function_type_checked_call_skip_convert, True, None),
        ])

    def test_additional(self) -> None:
        """Valid JSON string with an additional field."""
        self.do_test_function_with_targets(self.do_test_additional, [
            (Foo, False, None),
            (FooNamedTuple, False, None),
            (foo_function, False, None),
            (foo_function_type_checked_call, True, TypeError),
            (foo_function_type_checked_call_skip, True, None),
            (foo_function_type_checked_call_convert, True, TypeError),
            (foo_function_type_checked_call_skip_convert, True, None),
        ])

    def test_convert_ok(self) -> None:
        """Valid JSON string with an additional field."""
        self.do_test_function_with_targets(self.do_test_convert_ok, [
            (Foo, False, None),
            (FooNamedTuple, False, None),
            (foo_function, False, None),
            (foo_function_type_checked_call, True, TypeError),
            (foo_function_type_checked_call_skip, True, TypeError),
            (foo_function_type_checked_call_convert, True, None),
            (foo_function_type_checked_call_skip_convert, True, None),
        ])

    def test_additional_and_convert(self) -> None:
        """Valid JSON string with an additional field and one to convert."""
        self.do_test_function_with_targets(self.do_test_additional_and_convert, [
            (Foo, False, None),
            (FooNamedTuple, False, None),
            (foo_function, False, None),
            (foo_function_type_checked_call, True, TypeError),
            (foo_function_type_checked_call_skip, True, TypeError),
            (foo_function_type_checked_call_convert, True, TypeError),
            (foo_function_type_checked_call_skip_convert, True, None),
        ])

    def test_wrong_opt_type(self) -> None:
        """Valid JSON string with an additional field."""
        self.do_test_function_with_targets(self.do_test_wrong_opt_type, [
            (Foo, False, TypeError),
            (FooDecorated, True, TypeError),
            (FooNamedTuple, False, TypeError),
            (FooNamedTupleDecorated, True, TypeError),
            (foo_function, False, TypeError),
            (foo_function_type_checked_call, True, TypeError),
            (foo_function_type_checked_call_skip, True, TypeError),
            (foo_function_type_checked_call_convert, True, TypeError),
            (foo_function_type_checked_call_skip_convert, True, TypeError),
        ])

    def test_convert_error(self) -> None:
        """Valid JSON string with an additional field."""
        self.do_test_function_with_targets(self.do_test_convert_error, [
            (Foo, False, TypeError),
            (FooDecorated, True, TypeError),
            (FooNamedTuple, False, TypeError),
            (FooNamedTupleDecorated, True, TypeError),
            (foo_function, False, TypeError),
            (foo_function_type_checked_call, True, TypeError),
            (foo_function_type_checked_call_skip, True, TypeError),
            (foo_function_type_checked_call_convert, True, TypeError),
            (foo_function_type_checked_call_skip_convert, True, TypeError),
        ])

    def test_missing(self) -> None:
        """Invalid JSON string: missing a field."""
        self.do_test_function_with_targets(self.do_test_missing, [
            (Foo, False, TypeError),
            (FooDecorated, True, TypeError),
            (FooNamedTuple, False, TypeError),
            (FooNamedTupleDecorated, True, TypeError),
            (foo_function, False, TypeError),
            (foo_function_type_checked_call, True, TypeError),
            (foo_function_type_checked_call_skip, True, TypeError),
            (foo_function_type_checked_call_convert, True, TypeError),
            (foo_function_type_checked_call_skip_convert, True, TypeError),
        ])

    def test_opt_missing(self) -> None:
        """Valid JSON string without providing value for optional member."""
        self.do_test_function_with_targets(self.do_test_opt_missing, [
            (Foo, False, None),
            (FooDecorated, True, None),
            (FooNamedTuple, False, None),
            (FooNamedTupleDecorated, True, None),
            (foo_function, False, None),
            (foo_function_type_checked_call, True, None),
            (foo_function_type_checked_call_skip, True, None),
            (foo_function_type_checked_call_convert, True, None),
            (foo_function_type_checked_call_skip_convert, True, None),
        ])

    def test_incorrect_type(self) -> None:
        """Invalid JSON string: incorrect type of a field."""
        self.do_test_function_with_targets(self.do_test_incorrect_type, [
            (Foo, False, TypeError),
            (FooDecorated, True, TypeError),
            (FooNamedTuple, False, TypeError),
            (FooNamedTupleDecorated, True, TypeError),
            (foo_function, False, TypeError),
            (foo_function_type_checked_call, True, TypeError),
            (foo_function_type_checked_call_skip, True, TypeError),
            (foo_function_type_checked_call_convert, True, TypeError),
            (foo_function_type_checked_call_skip_convert, True, TypeError),
        ])


class TestErrorMessageOnIncorrectUsage(unittest.TestCase):
    """Avoid confusion"""

    def test_no_unpacking(self) -> None:
        """Should give meaningful error."""
        data = {
            "val": 42, "msg": "hello", "frac": 3.14, "flag": True, "opt": 10}
        with self.assertRaises(TypeError):
            foo_function_type_checked_call(data)  # pylint: disable=no-value-for-parameter


class TestUseOnDecorated(unittest.TestCase):
    """Tests call of type_checked_call on already decorated function."""

    def test_double_wrapping_function(self) -> None:
        """Should error to avoid confusion."""
        data = {
            "val": 42, "msg": "hello", "frac": 3.14, "flag": True, "opt": 10}
        with self.assertRaises(TypeError):
            type_checked_call()(foo_function_type_checked_call)(**data)

    def test_double_annotating_class(self) -> None:
        """Should also error"""
        with self.assertRaises(TypeError):
            @type_checked_constructor()  # pylint: disable=too-few-public-methods,unused-variable
            @type_checked_constructor()
            class DoubleTypeCheckedCtor:
                """Empty dummy"""
                pass  # pylint: disable=unnecessary-pass


class Point:  # pylint: disable=too-few-public-methods
    """Dummy point class."""

    def __init__(self, x_val: int, y_val: int) -> None:
        self.x_val: int = x_val
        self.y_val: int = y_val


class Speed:  # pylint: disable=too-few-public-methods
    """Dummy point class."""

    def __init__(self, x_val: int, y_val: int) -> None:
        self.x_val: int = x_val
        self.y_val: int = y_val


class Nested:  # pylint: disable=too-few-public-methods
    """Dummy class with a non-primitive member."""

    def __init__(self, pos: Point, opt_pos2: Optional[Point]) -> None:
        self.pos: Point = pos
        self.opt_pos2: Optional[Point] = opt_pos2


@type_checked_constructor()
class PointDecorated:  # pylint: disable=too-few-public-methods
    """Dummy point class."""

    def __init__(self, x_val: int, y_val: int) -> None:
        self.x_val: int = x_val
        self.y_val: int = y_val


@type_checked_constructor(skip=True)
class PointDecoratedSkip:  # pylint: disable=too-few-public-methods
    """Dummy point class."""

    def __init__(self, x_val: int, y_val: int) -> None:
        self.x_val: int = x_val
        self.y_val: int = y_val


@type_checked_constructor()
class NestedDecorated:  # pylint: disable=too-few-public-methods
    """Dummy class with a non-primitive member."""

    def __init__(self, pos: PointDecorated,
                 opt_pos2: Optional[PointDecorated],
                 pos_list: List[PointDecorated]) -> None:
        self.pos: PointDecorated = pos
        self.opt_pos2: Optional[PointDecorated] = opt_pos2
        self.pos_list: List[PointDecorated] = pos_list


@type_checked_constructor(convert=True)
class NestedDecoratedConvertPointSkip:  # pylint: disable=too-few-public-methods
    """Dummy class with a non-primitive member."""

    def __init__(self, pos: PointDecoratedSkip, val: int) -> None:
        self.pos: PointDecoratedSkip = pos
        self.val: int = val


@type_checked_constructor()  # pylint: disable=too-few-public-methods
class PointDecoratedNamedTuple(NamedTuple):
    """Dummy point class."""

    x_val: int
    y_val: int


@type_checked_constructor()  # pylint: disable=too-few-public-methods
class NestedDecoratedNamedTuple(NamedTuple):
    """Dummy class with a non-primitive member."""

    pos: PointDecoratedNamedTuple
    opt_pos2: Optional[PointDecoratedNamedTuple]
    pos_list: List[PointDecoratedNamedTuple]


class TestUnpackingNested(unittest.TestCase):
    """Tests with valid and invalid JSON strings."""

    def check_result(self, nested: Union[Nested,
                                         NestedDecorated,
                                         NestedDecoratedNamedTuple,
                                         NestedDecoratedConvertPointSkip]) -> None:
        """Validate content of Nested's members."""
        self.assertEqual(nested.pos.x_val, 1)
        self.assertEqual(nested.pos.y_val, 2)

    def test_ok(self) -> None:
        """Valid JSON string."""
        object_repr = '{"pos": {"x_val": 1, "y_val": 2}, "opt_pos2": {"x_val": 3, "y_val": 4}}'
        nested: Nested = type_checked_call()(Nested)(**json.loads(object_repr))
        self.check_result(nested)

    def test_superfluous_error_nested(self) -> None:
        """Should use plain ctor of nested and thus error."""
        object_repr = '''{"pos": {"x_val": 1, "y_val": 2, "too_much": 42},
                         "opt_pos2": {"x_val": 3, "y_val": 4}}'''
        with self.assertRaises(TypeError):
            type_checked_call()(Nested)(**json.loads(object_repr))

    def test_superfluous_error_opt_nested(self) -> None:  # pylint: disable=invalid-name
        """Should use plain ctor of nested and thus error."""
        object_repr = '''{"pos": {"x_val": 1, "y_val": 2},
                         "opt_pos2": {"x_val": 3, "y_val": 4, "too_much": 42}}'''
        with self.assertRaises(TypeError):
            type_checked_call()(Nested)(**json.loads(object_repr))

    def test_ok_decorated(self) -> None:
        """Valid JSON string."""
        object_repr = '''{
                "pos": {"x_val": 1, "y_val": 2},
                "opt_pos2": {"x_val": 3, "y_val": 4},
                "pos_list": [{"x_val": 3, "y_val": 4}]
            }'''
        nested_decorated: NestedDecorated = NestedDecorated(**json.loads(object_repr))
        self.check_result(nested_decorated)

    def test_ok_decorated_namedtuple(self) -> None:
        """Valid JSON string."""
        object_repr = '''{
                "pos": {"x_val": 1, "y_val": 2},
                "opt_pos2": {"x_val": 3, "y_val": 4},
                "pos_list": [{"x_val": 3, "y_val": 4}]
            }'''
        nested_decorated = NestedDecoratedNamedTuple(**json.loads(object_repr))
        self.check_result(nested_decorated)

    def test_ok_decorated_opt_none(self) -> None:
        """Valid JSON string."""
        object_repr = '{"pos": {"x_val": 1, "y_val": 2}, "opt_pos2": null, "pos_list": []}'
        nested_decorated: NestedDecorated = NestedDecorated(**json.loads(object_repr))
        self.check_result(nested_decorated)

    def test_ok_decorated_opt_missing(self) -> None:
        """Valid JSON string."""
        object_repr = '{"pos": {"x_val": 1, "y_val": 2}, "pos_list": []}'
        nested_decorated: NestedDecorated = NestedDecorated(**json.loads(object_repr))
        self.check_result(nested_decorated)

    def test_ok_nested_already_objects(self) -> None:
        """Valid JSON string."""
        object_dict = {"pos": Point(1, 2), "opt_pos2": Point(3, 4)}
        nested: Nested = type_checked_call()(Nested)(**object_dict)
        self.check_result(nested)

    def test_ok_nested_already_objects_wrong_type(self) -> None:  # pylint: disable=invalid-name
        """Valid JSON string."""
        object_dict = {"pos": Point(1, 2), "opt_pos2": Speed(3, 4)}
        with self.assertRaises(TypeError):
            type_checked_call()(Nested)(**object_dict)

    def test_ok_nested_decorated_objects(self) -> None:  # pylint: disable=invalid-name
        """Valid JSON string."""
        object_dict = {"pos": PointDecorated(1, 2),
                       "opt_pos2": PointDecorated(3, 4),
                       "pos_list": [PointDecorated(5, 6)]}
        nested_decorated: NestedDecorated = NestedDecorated(**object_dict)  # type: ignore
        self.check_result(nested_decorated)

    def test_ok_nested_objects_opt_none(self) -> None:
        """Valid JSON string."""
        object_dict = {"pos": Point(1, 2), "opt_pos2": None}
        nested: Nested = type_checked_call()(Nested)(**object_dict)
        self.check_result(nested)

    def test_ok_opt_null(self) -> None:
        """Valid JSON string with optional explicitly being None."""
        object_repr = '{"pos": {"x_val": 1, "y_val": 2}, "opt_pos2": null}'
        nested: Nested = type_checked_call()(Nested)(**json.loads(object_repr))
        self.check_result(nested)

    def test_ok_opt_missing(self) -> None:
        """Valid JSON string without optional field."""
        object_repr = '{"pos": {"x_val": 1, "y_val": 2}}'
        nested: Nested = type_checked_call()(Nested)(**json.loads(object_repr))
        self.check_result(nested)

    def test_from_dict_decorated(self) -> None:
        """Valid JSON string."""
        data = {"pos": PointDecorated(1, 2), "pos_list": []}
        nested: NestedDecorated = NestedDecorated(**data)  # type: ignore
        self.check_result(nested)

    def test_nested_decorated_differently(self) -> None:  # pylint: disable=invalid-name
        """Should use settings from inner class for inner ctor"""
        data = json.loads('{"pos": {"x_val": 1, "y_val": 2, "foo": 4}, "val": "3"}')
        nested: NestedDecoratedConvertPointSkip = \
            NestedDecoratedConvertPointSkip(**data)
        self.check_result(nested)

    def test_nested_decorated_differently_err_inner(self) -> None:  # pylint: disable=invalid-name
        """Should use settings from inner class for inner ctor"""
        data = json.loads('{"pos": {"x_val": 1, "y_val": "2"}, "val": 3}')
        with self.assertRaises(TypeError):
            NestedDecoratedConvertPointSkip(**data)

    def test_nested_decorated_differently_err_outer(self) -> None:  # pylint: disable=invalid-name
        """Should use settings from outer class for outer ctor"""
        data = json.loads('{"pos": {"x_val": 1, "y_val": 2}, "val": 3, "bar": 5}')
        with self.assertRaises(TypeError):
            NestedDecoratedConvertPointSkip(**data)


@type_checked_constructor()  # pylint: disable=too-few-public-methods
class Heart(NamedTuple):
    """Nested class"""
    weight_in_kg: float
    pulse_at_rest: int


@type_checked_constructor()  # pylint: disable=too-few-public-methods
class Human(NamedTuple):
    """Class having a nested member"""
    id: int
    name: str
    nick: Optional[str]
    heart: Heart
    friend_ids: List[int]


class TestUnpackingHuman(unittest.TestCase):
    """Tests with valid and invalid JSON strings."""

    def check_result(self, human: Human) -> None:
        """Validate content of Nested's members."""
        self.assertEqual(human.name, "Tobias")
        self.assertEqual(human.heart.pulse_at_rest, 52)

    @staticmethod
    def get_object_repr() -> str:
        """JSON string to decode"""
        return '''
            {
                "id": 1,
                "name": "Tobias",
                "heart": {
                    "weight_in_kg": 0.31,
                    "pulse_at_rest": 52
                },
                "friend_ids": [2, 3, 4, 5]
            }'''

    def test_ok(self) -> None:
        """Valid JSON string."""
        human: Human = Human(**json.loads(self.get_object_repr()))
        self.check_result(human)


class WithAny:  # pylint: disable=too-few-public-methods
    """Dummy class with an Amy member."""

    def __init__(self, val: Any) -> None:
        self.val: Any = val


class TestUnpackingWithAny(unittest.TestCase):
    """Tests with valid and invalid JSON strings."""

    def test_ok_str(self) -> None:
        """Valid JSON string."""
        object_repr = '{"val": "foo"}'
        with_any: WithAny = type_checked_call()(WithAny)(**json.loads(object_repr))
        self.assertEqual(with_any.val, "foo")

    def test_ok_float(self) -> None:
        """Valid JSON string."""
        object_repr = '{"val": 3.14}'
        with_any: WithAny = type_checked_call()(WithAny)(**json.loads(object_repr))
        self.assertEqual(with_any.val, 3.14)


class WithLists:  # pylint: disable=too-few-public-methods
    """Dummy class with a Union member."""

    def __init__(self,
                 ints: List[int],
                 opt_strs: List[Optional[str]],
                 opt_str_list: Optional[List[str]],
                 points: List[Point]) -> None:
        self.ints: List[int] = ints
        self.opt_strs: List[Optional[str]] = opt_strs
        self.opt_str_list: Optional[List[str]] = opt_str_list
        self.points: List[Point] = points


class TestUnpackingWithList(unittest.TestCase):
    """Tests with valid and invalid JSON strings."""

    def test_ok(self) -> None:
        """Valid JSON string"""
        object_repr = '{' \
                      '"ints": [1, 2],' \
                      '"opt_strs": ["a", "b"], ' \
                      '"opt_str_list": ["a", "b"], ' \
                      '"points": [{"x_val": 1, "y_val": 2}]}'
        with_list: WithLists = type_checked_call()(WithLists)(**json.loads(object_repr))
        self.assertEqual(with_list.ints, [1, 2])
        self.assertEqual(with_list.opt_strs, ["a", "b"])
        self.assertEqual(with_list.opt_str_list, ["a", "b"])
        self.assertEqual(with_list.points[0].x_val, 1)

    def test_ok_none(self) -> None:
        """Valid JSON string"""
        object_repr = '{' \
                      '"ints": [1, 2],' \
                      '"opt_strs": [null], ' \
                      '"opt_str_list": null, ' \
                      '"points": [{"x_val": 1, "y_val": 2}]}'
        with_list: WithLists = type_checked_call()(WithLists)(**json.loads(object_repr))
        self.assertEqual(with_list.ints, [1, 2])
        self.assertEqual(with_list.opt_strs, [None])
        self.assertEqual(with_list.opt_str_list, None)
        self.assertEqual(with_list.points[0].x_val, 1)

    def test_ok_empty(self) -> None:
        """Valid JSON string"""
        object_repr = '{' \
                      '"ints": [],' \
                      '"opt_strs": [], ' \
                      '"opt_str_list": [], ' \
                      '"points": []}'
        with_list: WithLists = type_checked_call()(WithLists)(**json.loads(object_repr))
        self.assertEqual(with_list.ints, [])
        self.assertEqual(with_list.opt_strs, [])
        self.assertEqual(with_list.opt_str_list, [])
        self.assertEqual(with_list.points, [])

    def test_incorrect_element_type(self) -> None:
        """Invalid JSON string."""
        object_repr = '{' \
                      '"ints": [1, 2.13],' \
                      '"opt_strs": ["a", "b"], ' \
                      '"opt_str_list": ["a", "b"], ' \
                      '"points": [{"x_val": 1, "y_val": 2}]}'
        with self.assertRaises(TypeError):
            type_checked_call()(WithLists)(**json.loads(object_repr))


class WithMemberFunc:  # pylint: disable=too-few-public-methods
    """Dummy class with a Union member."""

    def __init__(self, val_1: int, val_2: int) -> None:
        self.val_1: int = val_1
        self.val_2: int = val_2

    def member_func(self, msg: str) -> str:
        """Return value sum as string with message concatenated."""
        return str(self.val_1 + self.val_2) + msg


@type_checked_constructor()
class WithMemberFuncDecorated:  # pylint: disable=too-few-public-methods,unused-variable
    """Dummy class with a Union member."""

    def __init__(self, val_1: int, val_2: int) -> None:
        self.val_1: int = val_1
        self.val_2: int = val_2

    @type_checked_call()
    def member_func(self, msg: str) -> str:
        """Return value sum as string with message concatenated."""
        return str(self.val_1 + self.val_2) + msg


class TestUnpackingWithMemberFunc(unittest.TestCase):
    """Make sure member functions work too."""

    def test_ok(self) -> None:
        """Valid data dict."""
        data = {'val_1': 40, 'val_2': 2}
        with_member_func = WithMemberFunc(**data)
        result = type_checked_call()(with_member_func.member_func)('hello')
        self.assertEqual(result, '42hello')

    def test_invalid_type(self) -> None:
        """Incorrect type."""
        data = {'val_1': 40, 'val_2': 2}
        with_member_func = WithMemberFunc(**data)
        with self.assertRaises(TypeError):
            type_checked_call()(with_member_func.member_func)(3)

    def test_ok_decorated(self) -> None:
        """Valid data dict."""
        data = {'val_1': 40, 'val_2': 2}
        with_member_func = WithMemberFuncDecorated(**data)
        result = with_member_func.member_func('hello')
        self.assertEqual(result, '42hello')


class TestPickle(unittest.TestCase):
    """Annotated classes still need to work with pickle."""

    def test_dumps_and_reads(self) -> None:
        """Should work in both directions."""
        foo_obj = FooDecorated(1, "hi", True, None, 1.2)
        dump = pickle.dumps(foo_obj)
        foo_obj = pickle.loads(dump)
        self.assertEqual(1, foo_obj.val)


class WithUnion:  # pylint: disable=too-few-public-methods
    """Dummy class with a Union member."""

    def __init__(self, val: Union[int, str]) -> None:
        self.val: Union[int, str] = val


class TestUnpackingWithUnion(unittest.TestCase):
    """Make sure such classes are rejected."""

    def test_str(self) -> None:
        """Valid JSON string, but invalid target class."""
        object_repr = '{"val": "hi"}'
        with self.assertRaises(TypeError):
            type_checked_call()(WithUnion)(**json.loads(object_repr))


class WithoutTypeAnnotation:  # pylint: disable=too-few-public-methods
    """Dummy class with a non-annotated type."""

    def __init__(self, val) -> None:  # type: ignore
        self.val = val


class TestUnpackingWithoutTypeAnnotation(unittest.TestCase):
    """Make sure such classes are rejected."""

    def test_str(self) -> None:
        """Valid JSON string, but invalid target class."""
        object_repr = '{"val": "hi"}'
        with self.assertRaises(TypeError):
            type_checked_call()(WithoutTypeAnnotation)(**json.loads(object_repr))


class WithDict:  # pylint: disable=too-few-public-methods
    """Dummy class with a Dict member."""

    def __init__(self, val: Dict[int, str]) -> None:
        self.val: Dict[int, str] = val


class TestUnpackingWithDict(unittest.TestCase):
    """Make sure such classes are rejected."""

    def test_str(self) -> None:
        """Valid JSON string, but invalid target class."""
        object_repr = '{"val": {"key1": 1, "key2": 2}}'
        with self.assertRaises(TypeError):
            type_checked_call()(WithDict)(**json.loads(object_repr))


class WithArgs:  # pylint: disable=too-few-public-methods
    """Dummy class with a Union member."""

    def __init__(self, *args: int) -> None:
        self.args: Any = args


class TestUnpackingWithArgs(unittest.TestCase):
    """Make sure such classes are rejected."""

    def test_str(self) -> None:
        """Invalid target class."""
        object_repr = '{}'
        with self.assertRaises(TypeError):
            type_checked_call()(WithArgs)(**json.loads(object_repr))


class WithKWArgs:  # pylint: disable=too-few-public-methods
    """Dummy class with a Union member."""

    def __init__(self, **kwargs: int) -> None:
        self.kwargs: Any = kwargs


class TestUnpackingWithKWArgs(unittest.TestCase):
    """Make sure such classes are rejected."""

    def test_str(self) -> None:
        """Invalid target class."""
        object_repr = '{}'
        with self.assertRaises(TypeError):
            type_checked_call()(WithKWArgs)(**json.loads(object_repr))


class TestUnpackingToNonCallable(unittest.TestCase):
    """Make sure such things are rejected."""

    def test_str(self) -> None:
        """Invalid target."""
        object_repr = '{}'
        with self.assertRaises(TypeError):
            type_checked_call()('hi')(**json.loads(object_repr))  # type: ignore


class WithDefault(NamedTuple):  # pylint: disable=too-few-public-methods
    """Some dummy class as a NamedTuple with one default value."""
    val: int
    msg: str = "hi"


class TestWithDefault(unittest.TestCase):
    """Make sure default values to not need to be provided."""

    def test_all_provided(self) -> None:
        """Should be OK."""
        obj = type_checked_call()(WithDefault)(val=42, msg='hello')
        self.assertEqual(obj.val, 42)
        self.assertEqual(obj.msg, 'hello')

    def test_only_needed_provided(self) -> None:
        """Should be OK."""
        obj = type_checked_call()(WithDefault)(val=42)
        self.assertEqual(obj.val, 42)
        self.assertEqual(obj.msg, 'hi')
