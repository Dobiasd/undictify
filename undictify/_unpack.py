"""
undictify - Type-safe dictionary unpacking / JSON deserialization
"""

import inspect
import json
from typing import Any, Callable, Dict, List
from typing import Type, TypeVar
from typing import _Union  # type: ignore

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
                            f'{__get_type_name(json_type)} instead of '
                            f'{__get_type_name(target_type)}.')

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
    """Return all types a Union can hold."""
    assert __is_union_type(union_type)
    return union_type.__args__  # type: ignore


def __get_optional_type(optional_type: Type[TypeT]) -> Type[Any]:
    """Return the type an Optional can hold."""
    assert __is_optional_type(optional_type)
    args = optional_type.__args__  # type: ignore
    assert len(args) == 2
    return args[0]  # type: ignore


def __get_type_name(the_type: Type[TypeT]) -> str:
    """Return a printable name of a type."""
    if __is_optional_type(the_type):
        return f'Optional[{str(__get_optional_type(the_type).__name__)}]'
    if __is_list_type(the_type):
        return f'List[{str(__get_list_type_elem_type(the_type).__name__)}]'
    return the_type.__name__


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
