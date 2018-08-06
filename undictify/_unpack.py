"""
undictify - Type-checked function calls at runtime
"""

import inspect
import sys
from typing import Any, Callable, Dict, List, Type, TypeVar, Union

VER_3_7_AND_UP = sys.version_info[:3] >= (3, 7, 0)  # PEP 560

# pylint: disable=no-name-in-module
if VER_3_7_AND_UP:
    from typing import _GenericAlias  # type: ignore
else:
    from typing import _Union  # type: ignore
# pylint: enable=no-name-in-module

TypeT = TypeVar('TypeT')


def type_checked_call(func: Callable[..., TypeT]) -> Callable[..., TypeT]:
    """Decorator that type checks arguments to every call of a function."""

    def __undictify_wrapper_func(*args: Any, **kwargs: Any) -> TypeT:
        return type_checked_apply(func, *args, **kwargs)

    __undictify_wrapper_func = __set_undictify_wrapped_func(__undictify_wrapper_func, func)
    return __undictify_wrapper_func


def type_checked_call_skip(func: Callable[..., TypeT]) -> Callable[..., TypeT]:
    """Decorator that type checks arguments to every call of a function.
    It skips all keyword arguments that the function does not take."""

    def __undictify_wrapper_func(*args: Any, **kwargs: Any) -> TypeT:
        return type_checked_apply_skip(func, *args, **kwargs)

    __undictify_wrapper_func = __set_undictify_wrapped_func(__undictify_wrapper_func, func)
    return __undictify_wrapper_func


def type_checked_call_convert(func: Callable[..., TypeT]) -> Callable[..., TypeT]:
    """Decorator that type checks arguments to every call of a function.
    It converts arguments into target types of parameters if possible."""

    def __undictify_wrapper_func(*args: Any, **kwargs: Any) -> TypeT:
        return type_checked_apply_convert(func, *args, **kwargs)

    __undictify_wrapper_func = __set_undictify_wrapped_func(__undictify_wrapper_func, func)
    return __undictify_wrapper_func


def type_checked_call_skip_convert(func: Callable[..., TypeT]) -> Callable[..., TypeT]:
    """Decorator that type checks arguments to every call of a function.
    It skips all keyword arguments that the function does not take and
    converts arguments into target types of parameters if possible."""

    def __undictify_wrapper_func(*args: Any, **kwargs: Any) -> TypeT:
        return type_checked_apply_skip_convert(func, *args, **kwargs)

    __undictify_wrapper_func = __set_undictify_wrapped_func(__undictify_wrapper_func, func)
    return __undictify_wrapper_func


def type_checked_apply(func: Callable[..., TypeT],
                       *args: Any, **kwargs: Any) -> TypeT:
    """Type check the arguments of a function call."""
    return __unpack_dict(func, __merge_args_and_kwargs(func, *args, **kwargs),
                         False, False)


def type_checked_apply_skip(func: Callable[..., TypeT],
                            *args: Any, **kwargs: Any) -> TypeT:
    """Type check the arguments of a function call.
    Skips all keyword arguments that the function does not take."""
    return __unpack_dict(func, __merge_args_and_kwargs(func, *args, **kwargs),
                         True, False)


def type_checked_apply_convert(func: Callable[..., TypeT],
                               *args: Any, **kwargs: Any) -> TypeT:
    """Type check the arguments of a function call.
    Convert arguments into target types of parameters if possible."""
    return __unpack_dict(func, __merge_args_and_kwargs(func, *args, **kwargs),
                         False, True)


def type_checked_apply_skip_convert(func: Callable[..., TypeT],
                                    *args: Any, **kwargs: Any) -> TypeT:
    """Decorator that type checks arguments to every call of a function.
    Skips all keyword arguments that the function does not take and
    convert arguments into target types of parameters if possible."""
    return __unpack_dict(func, __merge_args_and_kwargs(func, *args, **kwargs),
                         True, True)


def __merge_args_and_kwargs(func: Callable[..., TypeT],
                            *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Returns one kwargs dictionary or
    raises an exeption in case of overlapping-name problems."""
    signature = inspect.signature(func)
    param_names = [param.name for param in signature.parameters.values()]
    if len(args) > len(param_names):
        raise ValueError(f'Too many parameters for {func.__name__}.')
    args_as_kwargs = dict(zip(param_names, list(args)))
    keys_in_args_and_kwargs = set.intersection(set(args_as_kwargs.keys()),
                                               set(kwargs.keys()))
    if keys_in_args_and_kwargs:
        raise ValueError(f'The following parameters are given as '
                         f'arg and kwarg in call of {func.__name__}: '
                         f'{keys_in_args_and_kwargs}')

    return {**args_as_kwargs, **kwargs}


def __unpack_dict(func: Callable[..., TypeT],
                  data: Dict[str, Any],
                  skip_superfluous: bool,
                  convert_types: bool) -> TypeT:
    """Constructs an object in a type-safe way from a dictionary."""

    if not callable(func):
        raise TypeError(f'Target "{func}" is not callable.')

    assert __is_dict(data), 'Argument data needs to be a dictionary.'

    if __is_undictify_wrapped_func(func):
        return func(**data)

    signature = inspect.signature(func)
    ctor_params: Dict[str, Any] = {}

    if not skip_superfluous:
        param_names = [param.name for param in signature.parameters.values()]
        argument_names = data.keys()
        superfluous = set(argument_names) - set(param_names)
        if superfluous:
            raise ValueError(f'Superfluous parameters in call: {superfluous}')

    for param in signature.parameters.values():
        if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise TypeError('Only parameters of kind POSITIONAL_OR_KEYWORD '
                            'supported in target functions.')
        if __is_union_type(param.annotation) \
                and not __is_optional_type(param.annotation):
            raise TypeError('Union members in target function other than Optional '
                            'are not supported.')
        if __is_dict_type(param.annotation):
            raise TypeError('Dict members in target function are not supported.')
        if param.name not in data:
            if __is_optional_type(param.annotation):
                ctor_params[param.name] = None
            else:
                raise ValueError(f'Key {param.name} is missing.')
        else:
            ctor_params[param.name] = __get_value(param.annotation,
                                                  data[param.name],
                                                  param.name,
                                                  skip_superfluous,
                                                  convert_types)

    return func(**ctor_params)


def __get_value(target_type: Callable[..., TypeT], value: Any, log_name: str,
                skip_superfluous: bool, convert_types: bool) -> Any:
    """Convert a single value into target type if possible."""
    if __is_list(value):
        return __get_list_value(target_type, value, log_name,
                                skip_superfluous, convert_types)

    if __is_dict(value):
        return __get_dict_value(target_type, value,
                                skip_superfluous, convert_types)

    if __is_undictify_wrapped_func(target_type):
        return __get_undictify_wrapped_func_value(target_type, value)

    allowed_types = list(map(__unwrap_decorator_type, __get_union_types(target_type) \
        if __is_optional_type(target_type) \
        else [target_type]))

    if target_type is inspect.Parameter.empty:
        raise TypeError(f'Parameter {log_name} of target function '
                        'is missing a type annotation.')

    if Any not in allowed_types:
        if not __isinstanceofone(value, allowed_types):
            value_type = type(value)
            if convert_types:
                if __is_optional_type(target_type):
                    target_type = __get_optional_type(target_type)
                try:
                    return target_type(value)
                except ValueError:
                    raise TypeError(f'Can not convert {value} '
                                    f'from type {__get_type_name(value_type)} '
                                    f'into type {__get_type_name(target_type)} '
                                    f'for key {log_name}.')

            raise TypeError(f'Key {log_name} has incorrect type: '
                            f'{__get_type_name(value_type)} instead of '
                            f'{__get_type_name(target_type)}.')

    return value


def __get_list_value(target_type: Callable[..., TypeT], value: Any,
                     log_name: str,
                     skip_superfluous: bool, convert_types: bool) -> Any:
    if not __is_list_type(target_type) and \
            not __is_optional_list_type(target_type):
        raise TypeError(f'No list expected for {log_name}')
    target_elems = []
    target_elem_type = __get_list_type_elem_type(target_type)
    for elem in value:
        target_elems.append(__get_value(target_elem_type,
                                        elem, value,
                                        skip_superfluous, convert_types))
    return target_elems


def __get_dict_value(target_type: Callable[..., TypeT], value: Any,
                     skip_superfluous: bool, convert_types: bool) -> Any:
    if __is_optional_type(target_type):
        return __unpack_dict(__get_optional_type(target_type),
                             value, skip_superfluous, convert_types)
    return __unpack_dict(target_type, value, skip_superfluous, convert_types)


def __get_undictify_wrapped_func_value(target_type: Callable[..., TypeT],
                                       value: Any) -> Any:
    if __is_dict(value):
        return target_type(**value)
    wrapped_function = __get_undictify_wrapped_func(target_type)
    if isinstance(value, wrapped_function):  # type: ignore
        return value
    return wrapped_function(value)


def __is_union_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a Union."""
    if VER_3_7_AND_UP:
        return (the_type is Union or  # pylint: disable=unidiomatic-typecheck
                isinstance(the_type, _GenericAlias) and the_type.__origin__ is Union)
    return type(the_type) is _Union  # pylint: disable=unidiomatic-typecheck


def __is_list_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a List."""
    try:
        if VER_3_7_AND_UP:
            return isinstance(the_type,
                              _GenericAlias) and the_type.__origin__ is list
        return issubclass(the_type, List)  # type: ignore
    except TypeError:
        return False


def __is_optional_list_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a Optional[List]."""
    if __is_list_type(the_type):
        return True
    if __is_optional_type(the_type) and \
            __is_list_type(__get_optional_type(the_type)):
        return True
    return False


def __is_dict_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a Dict."""
    try:
        if VER_3_7_AND_UP:
            return isinstance(the_type,
                              _GenericAlias) and the_type.__origin__ is dict
        return issubclass(the_type, Dict)  # type: ignore
    except TypeError:
        return False


def __get_union_types(union_type: Callable[..., TypeT]) -> List[Callable[..., TypeT]]:
    """Return all types a Union can hold."""
    assert __is_union_type(union_type)
    return union_type.__args__  # type: ignore


def __get_optional_type(optional_type: Callable[..., TypeT]) -> Type[Any]:
    """Return the type an Optional can hold."""
    assert __is_optional_type(optional_type)
    args = optional_type.__args__  # type: ignore
    assert len(args) == 2
    return args[0]  # type: ignore


def __get_type_name(the_type: Callable[..., TypeT]) -> str:
    """Return a printable name of a type."""
    if __is_optional_type(the_type):
        return f'Optional[{str(__get_optional_type(the_type).__name__)}]'
    if __is_list_type(the_type):
        return f'List[{str(__get_list_type_elem_type(the_type).__name__)}]'
    return the_type.__name__


def __get_list_type_elem_type(list_type: Callable[..., TypeT]) -> Callable[..., Any]:
    """Return the type of a single element of the list type."""
    if __is_optional_type(list_type):
        list_type = __get_optional_type(list_type)
    assert __is_list_type(list_type)
    list_args = list_type.__args__  # type: ignore
    assert len(list_args) == 1
    return list_args[0]  # type: ignore


def __isinstanceofone(value: Callable[..., TypeT], types: List[Callable[..., TypeT]]) -> bool:
    """Check if value is an instance of one of the given types."""
    for the_type in types:
        if __is_union_type(the_type):
            if __isinstanceofone(value, __get_union_types(the_type)):
                return True
        try:
            if isinstance(value, the_type):  # type: ignore
                return True
        except TypeError:
            pass
    return False


def __is_primitive(value: Callable[..., TypeT]) -> bool:
    """Check if int, str, bool, float or None."""
    return isinstance(value, (int, str, bool, float, type(None)))


def __is_optional_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is an Optional."""
    if not __is_union_type(the_type):
        return False
    union_args = __get_union_types(the_type)
    return len(union_args) == 2 and __isinstanceofone(None, union_args)  # type: ignore


def __is_dict(value: TypeT) -> bool:
    """Return True if the value is a dictionary."""
    return isinstance(value, dict)


def __is_list(value: TypeT) -> bool:
    """Return True if the value is a list."""
    return isinstance(value, list)


def __unwrap_decorator_type(the_type: Callable[..., TypeT]) -> Callable[..., Any]:
    """Get the actual type returned by the internal wrapper"""
    try:
        if __is_undictify_wrapped_func(the_type):
            return __get_undictify_wrapped_func(the_type)
        return the_type
    except TypeError:
        return the_type


def __is_undictify_wrapped_func(the_type: Callable[..., TypeT]) -> bool:
    return hasattr(the_type, '__undictify_wrapped_func')  # pylint: disable=protected-access


def __get_undictify_wrapped_func(the_type: Callable[..., TypeT]) -> Callable[..., Any]:
    assert __is_undictify_wrapped_func(the_type)
    # pylint: disable=protected-access
    result = the_type.__undictify_wrapped_func  #  type: ignore
    # pylint: enable=protected-access
    return result #  type: ignore


def __set_undictify_wrapped_func(wrapping_func: Callable[..., TypeT],
                                 wrapped_func: Callable[..., Any]) -> Callable[..., TypeT]:
    # pylint: disable=protected-access
    wrapping_func.__undictify_wrapped_func = wrapped_func #  type: ignore
    # pylint: enable=protected-access
    return wrapping_func
