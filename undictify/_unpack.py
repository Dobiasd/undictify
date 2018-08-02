#!/usr/bin/env python3

"""
undictify - Type-safe dictionary unpacking / JSON deserialization
"""

import inspect
import json
import unittest
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union
from typing import Type, TypeVar
from typing import _Union  # type: ignore

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2018, Tobias Hermann"
__email__ = "editgym@gmail.com"
__license__ = "MIT"

TypeT = TypeVar('TypeT')


def unpack_dict(target_func: Callable[..., TypeT],
                data: Dict[str, Any],
                convert_types: bool = False) -> TypeT:
    """Constructs an object in a type-safe way from a dictionary."""

    if not callable(target_func):
        raise TypeError(f'Target "{target_func}" is not callable.')

    signature = inspect.signature(target_func)
    ctor_params: Dict[str, Any] = {}

    for param in signature.parameters.values():
        if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise TypeError('Only parameters of kind POSITIONAL_OR_KEYWORD '
                            'supported in target classes.')
        if __is_union_type(param.annotation) \
                and not __is_optional_type(param.annotation):
            raise TypeError('Union members in target class other than Optional '
                            'are not supported.')
        if __is_dict_type(param.annotation):
            raise TypeError('Dict members in target class are not supported.')
        if param.name not in data:
            if __is_optional_type(param.annotation):
                ctor_params[param.name] = None
            else:
                raise ValueError(f'Key {param.name} is missing.')
        else:
            ctor_params[param.name] = __get_value(param.annotation,
                                                  data[param.name],
                                                  param.name,
                                                  convert_types)

    return target_func(**ctor_params)


def unpack_json(target_func: Callable[..., TypeT],
                object_repr: str, convert_types: bool = False) -> TypeT:
    """Constructs an object in a type-safe way from a JSON strings."""
    return unpack_dict(target_func, json.loads(object_repr),
                       convert_types)


def __get_value(target_type: Type[TypeT], value: Any,
                log_name: str, convert_types: bool = False) -> Any:
    """Convert a single value into target type if possible."""
    if __is_list(value):
        if not __is_list_type(target_type) and \
                not __is_optional_list_type(target_type):
            raise ValueError(f'No list expected for {log_name}')
        target_elems = []
        target_elem_type = __get_list_type_elem_type(target_type)
        for elem in value:
            target_elems.append(__get_value(target_elem_type,
                                            elem, value))
        return target_elems

    if __is_dict(value):
        return unpack_dict(target_type, value, convert_types)

    if __is_optional_type(target_type):
        allowed_types = __get_union_types(target_type)
    else:
        allowed_types = [target_type]

    if target_type is inspect.Parameter.empty:
        raise TypeError(f'Parameter {log_name} of target class '
                        'is missing a type annotation.')

    if Any not in allowed_types:
        if not __isinstanceofone(value, allowed_types):
            json_type = type(value)
            if convert_types:
                if __is_optional_type(target_type):
                    return __get_optional_type(target_type)(value)
                return target_type(value)  # type: ignore
            raise TypeError(f'Key {log_name} has incorrect type: '
                            f'{json_type.__name__} instead of '
                            f'{target_type.__name__}.')

    return value


def __is_union_type(the_type: Type[TypeT]) -> bool:
    """Return True if the type is a Union."""
    return type(the_type) is _Union  # pylint: disable=unidiomatic-typecheck


def __is_list_type(the_type: Type[TypeT]) -> bool:
    """Return True if the type is a List."""
    try:
        return issubclass(the_type, List)
    except TypeError:
        return False


def __is_optional_list_type(the_type: Type[TypeT]) -> bool:
    """Return True if the type is a Optional[List]."""
    if __is_list_type(the_type):
        return True
    if __is_optional_type(the_type) and \
            __is_list_type(__get_optional_type(the_type)):
        return True
    return False


def __is_dict_type(the_type: Type[TypeT]) -> bool:
    """Return True if the type is a Dict."""
    try:
        return issubclass(the_type, Dict)
    except TypeError:
        return False


def __get_union_types(union_type: Type[TypeT]) -> List[Type[Any]]:
    assert __is_union_type(union_type)
    return union_type.__args__  # type: ignore


def __get_optional_type(optional_type: Type[TypeT]) -> Type[Any]:
    assert __is_optional_type(optional_type)
    args = optional_type.__args__  # type: ignore
    assert len(args) == 2
    return args[0]  # type: ignore


def __get_list_type_elem_type(list_type: Type[TypeT]) -> Type[TypeT]:
    """Return the type of a single element of the list type."""
    if __is_optional_type(list_type):
        list_type = __get_optional_type(list_type)
    assert __is_list_type(list_type)
    list_args = list_type.__args__  # type: ignore
    assert len(list_args) == 1
    return list_args[0]  # type: ignore


def __isinstanceofone(value: TypeT, types: List[Type[Any]]) -> bool:
    """Check if value is an instance of one of the given types."""
    for the_type in types:
        if __is_union_type(the_type):
            if __isinstanceofone(value, __get_union_types(the_type)):
                return True
        try:
            if isinstance(value, the_type):
                return True
        except TypeError:
            pass
    return False


def __is_primitive(value: TypeT) -> bool:
    """Check if int, str, bool, float or None."""
    return isinstance(value, (int, str, bool, float, type(None)))


def __is_optional_type(the_type: Type[TypeT]) -> bool:
    """Return True if the type is an Optional."""
    if not __is_union_type(the_type):
        return False
    union_args = __get_union_types(the_type)
    return len(union_args) == 2 and __isinstanceofone(None, union_args)


def __is_dict(value: TypeT) -> bool:
    """Return True if the value is a dictionary."""
    return isinstance(value, dict)


def __is_list(value: TypeT) -> bool:
    """Return True if the value is a list."""
    return isinstance(value, list)


class Foo:  # pylint: disable=too-few-public-methods
    """Some dummy class."""

    def __init__(self,  # pylint: disable=too-many-arguments,line-too-long
                 val: int, msg: str, flag: bool, opt: Optional[int],
                 frac: float = 1.23) -> None:
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


def foo_function(val: int, msg: str, frac: float, flag: bool,
                 opt: Optional[int]) -> Foo:
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

    def do_test_dict(self, target_func: Callable[..., TypeT]) -> None:
        """Valid data dict."""
        data = {
            "val": 42, "msg": "hello", "frac": 3.14, "flag": True, "opt": 10}
        a_foo = unpack_dict(target_func, data)
        self.check_result(a_foo, 10)

    def do_test_ok(self, target_func: Callable[..., TypeT]) -> None:
        """Valid JSON string."""
        object_repr = '{"val": 42, "msg": "hello", "frac": 3.14, ' \
                      '"flag": true, "opt": 10}'
        a_foo = unpack_json(target_func, object_repr)
        self.check_result(a_foo, 10)

    def do_test_opt_null(self, target_func: Callable[..., TypeT]) -> None:
        """Valid JSON string null for the optional member."""
        object_repr = '{"val": 42, "msg": "hello", "frac": 3.14, ' \
                      '"flag": true, "opt": null}'
        a_foo = unpack_json(target_func, object_repr)
        self.check_result(a_foo, None)

    def do_test_additional(self, target_func: Callable[..., TypeT]) -> None:
        """Valid JSON string with an additional field."""
        object_repr = '{"val": 42, "msg": "hello", "frac": 3.14, ''' \
                      '"flag": true, "opt": 10, "ignore": 1}'
        a_foo = unpack_json(target_func, object_repr)
        self.check_result(a_foo, 10)

    def do_test_convert_ok(self, target_func: Callable[..., TypeT]) -> None:
        """Valid JSON string."""
        object_repr = '{"val": "42", "msg": 5, "frac": 3, ' \
                      '"flag": true, "opt": 10.1}'
        a_foo = unpack_json(target_func, object_repr, True)
        self.check_result(a_foo, 10, 3.0, '5')

    def do_test_convert_error(self, target_func: Callable[..., TypeT]) -> None:
        """Valid JSON string."""
        object_repr = '{"val": "twentyfour", "msg": "hello", "frac": 3.14, ' \
                      '"flag": true, "opt": 10}'
        with self.assertRaises(ValueError):
            unpack_json(target_func, object_repr, True)

    def do_test_missing(self, target_func: Callable[..., TypeT]) -> None:
        """Invalid JSON string: missing a field."""
        object_repr = '{"val": 42, "msg": "hello", "opt": 10, "flag": true}'
        with self.assertRaises(ValueError):
            unpack_json(target_func, object_repr)

    def do_test_opt_missing(self, target_func: Callable[..., TypeT]) -> None:
        """Valid JSON string without providing value for optional member."""
        object_repr = '{"val": 42, "msg": "hello", "frac": 3.14, "flag": true}'
        a_foo = unpack_json(target_func, object_repr)
        self.check_result(a_foo, None)

    def do_test_incorrect_type(self, target_func: Callable[..., TypeT]) -> None:
        """Invalid JSON string: incorrect type of a field."""
        object_repr = '{"val": 42, "msg": "hello", "opt": 10, ' \
                      '"frac": "incorrect", "flag": true}'
        with self.assertRaises(TypeError):
            unpack_json(target_func, object_repr)

    def do_test_invalid_json(self, target_func: Callable[..., TypeT]) -> None:
        """Invalid JSON string: broken format."""
        object_repr = 'I am not a JSON string'
        with self.assertRaises(json.decoder.JSONDecodeError):
            unpack_json(target_func, object_repr)

    def test_dict(self) -> None:
        """Valid data dict."""
        self.do_test_dict(Foo)
        self.do_test_dict(FooNamedTuple)
        self.do_test_dict(foo_function)

    def test_ok(self) -> None:
        """Valid JSON string."""
        self.do_test_ok(Foo)
        self.do_test_ok(FooNamedTuple)
        self.do_test_ok(foo_function)

    def test_opt_null(self) -> None:
        """Valid JSON string null for the optional member."""
        self.do_test_opt_null(Foo)
        self.do_test_opt_null(FooNamedTuple)
        self.do_test_opt_null(foo_function)

    def test_additional(self) -> None:
        """Valid JSON string with an additional field."""
        self.do_test_additional(Foo)
        self.do_test_additional(FooNamedTuple)
        self.do_test_additional(foo_function)

    def test_convert_ok(self) -> None:
        """Valid JSON string with an additional field."""
        self.do_test_convert_ok(Foo)
        self.do_test_convert_ok(FooNamedTuple)
        self.do_test_convert_ok(foo_function)

    def test_convert_error(self) -> None:
        """Valid JSON string with an additional field."""
        self.do_test_convert_error(Foo)
        self.do_test_convert_error(FooNamedTuple)
        self.do_test_convert_error(foo_function)

    def test_missing(self) -> None:
        """Invalid JSON string: missing a field."""
        self.do_test_missing(Foo)
        self.do_test_missing(FooNamedTuple)
        self.do_test_missing(foo_function)

    def test_opt_missing(self) -> None:
        """Valid JSON string without providing value for optional member."""
        self.do_test_opt_missing(Foo)
        self.do_test_opt_missing(FooNamedTuple)
        self.do_test_opt_missing(foo_function)

    def test_incorrect_type(self) -> None:
        """Invalid JSON string: incorrect type of a field."""
        self.do_test_incorrect_type(Foo)
        self.do_test_incorrect_type(FooNamedTuple)
        self.do_test_incorrect_type(foo_function)

    def test_invalid_json(self) -> None:
        """Invalid JSON string: broken format."""
        self.do_test_invalid_json(Foo)
        self.do_test_invalid_json(FooNamedTuple)
        self.do_test_invalid_json(foo_function)


class Point:  # pylint: disable=too-few-public-methods
    """Dummy point class."""

    def __init__(self, x_val: int, y_val: int) -> None:
        self.x_val: int = x_val
        self.y_val: int = y_val


class Nested:  # pylint: disable=too-few-public-methods
    """Dummy class with a non-primitive member."""

    def __init__(self, pos: Point) -> None:
        self.pos: Point = pos


class TestUnpackingNested(unittest.TestCase):
    """Tests with valid and invalid JSON strings."""

    def check_result(self, nested: Nested) -> None:
        """Validate content of Nested's members."""
        self.assertEqual(nested.pos.x_val, 1)
        self.assertEqual(nested.pos.y_val, 2)

    def test_ok(self) -> None:
        """Valid JSON string."""
        object_repr = '{"pos": {"x_val": 1, "y_val": 2}}'
        nested: Nested = unpack_json(Nested, object_repr)
        self.check_result(nested)

    def test_from_dict(self) -> None:
        """Valid JSON string."""
        data = {"pos": Point(1, 2)}
        nested: Nested = unpack_dict(Nested, data)
        self.check_result(nested)


class WithAny:  # pylint: disable=too-few-public-methods
    """Dummy class with an Amy member."""

    def __init__(self, val: Any) -> None:
        self.val: Any = val


class TestUnpackingWithAny(unittest.TestCase):
    """Tests with valid and invalid JSON strings."""

    def test_ok_str(self) -> None:
        """Valid JSON string."""
        object_repr = '{"val": "foo"}'
        with_any: WithAny = unpack_json(WithAny, object_repr)
        self.assertEqual(with_any.val, "foo")

    def test_ok_float(self) -> None:
        """Valid JSON string."""
        object_repr = '{"val": 3.14}'
        with_any: WithAny = unpack_json(WithAny, object_repr)
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
        with_list: WithLists = unpack_json(WithLists, object_repr)
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
        with_list: WithLists = unpack_json(WithLists, object_repr)
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
        with_list: WithLists = unpack_json(WithLists, object_repr)
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
            unpack_json(WithLists, object_repr)


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
            unpack_json(WithUnion, object_repr)


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
            unpack_json(WithoutTypeAnnotation, object_repr)


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
            unpack_json(WithDict, object_repr)


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
            unpack_json(WithArgs, object_repr)


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
            unpack_json(WithKWArgs, object_repr)


class TestUnpackingToNonCallable(unittest.TestCase):
    """Make sure such things are rejected."""

    def test_str(self) -> None:
        """Invalid target."""
        object_repr = '{}'
        with self.assertRaises(TypeError):
            unpack_json("hi", object_repr)  # type: ignore
