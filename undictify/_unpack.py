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

    def wrapper(*args: Any, **kwargs: Any) -> TypeT:
        return _unpack_dict(func,  # type: ignore
                            _merge_args_and_kwargs(func, *args, **kwargs),
                            False, False)

    setattr(wrapper, '_undictify_wrapped_func', func)
    return wrapper


def type_checked_call_skip(func: Callable[..., TypeT]) -> Callable[..., TypeT]:
    """Decorator that type checks arguments to every call of a function.
    It skips all keyword arguments that the function does not take."""

    def wrapper(*args: Any, **kwargs: Any) -> TypeT:
        return _unpack_dict(func,  # type: ignore
                            _merge_args_and_kwargs(func, *args, **kwargs),
                            True, False)

    setattr(wrapper, '_undictify_wrapped_func', func)
    return wrapper


def type_checked_call_convert(func: Callable[..., TypeT]) -> Callable[..., TypeT]:
    """Decorator that type checks arguments to every call of a function.
    It converts arguments into target types of parameters if possible."""

    def wrapper(*args: Any, **kwargs: Any) -> TypeT:
        return _unpack_dict(func,  # type: ignore
                            _merge_args_and_kwargs(func, *args, **kwargs),
                            False, True)

    setattr(wrapper, '_undictify_wrapped_func', func)
    return wrapper


def type_checked_call_skip_convert(func: Callable[..., TypeT]) -> Callable[..., TypeT]:
    """Decorator that type checks arguments to every call of a function.
    It skips all keyword arguments that the function does not take and
    converts arguments into target types of parameters if possible."""

    def wrapper(*args: Any, **kwargs: Any) -> TypeT:
        return _unpack_dict(func,  # type: ignore
                            _merge_args_and_kwargs(func, *args, **kwargs),
                            True, True)

    setattr(wrapper, '_undictify_wrapped_func', func)
    return wrapper


WrappedOrFunc = Callable[..., TypeT]


def _is_wrapped_func(func: WrappedOrFunc) -> bool:
    return hasattr(func, '_undictify_wrapped_func')


def type_checked_apply(func: WrappedOrFunc,
                       *args: Any, **kwargs: Any) -> Any:
    """Type check the arguments of a function call."""
    return type_checked_call(func)(*args, **kwargs)


def type_checked_apply_skip(func: WrappedOrFunc,
                            *args: Any, **kwargs: Any) -> Any:
    """Type check the arguments of a function call.
    Skips all keyword arguments that the function does not take."""
    return type_checked_call_skip(func)(*args, **kwargs)


def type_checked_apply_convert(func: WrappedOrFunc,
                               *args: Any, **kwargs: Any) -> Any:
    """Type check the arguments of a function call.
    Convert arguments into target types of parameters if possible."""
    return type_checked_call_convert(func)(*args, **kwargs)


def type_checked_apply_skip_convert(func: WrappedOrFunc,
                                    *args: Any, **kwargs: Any) -> Any:
    """Type check the arguments of a function call.
    Skips all keyword arguments that the function does not take and
    convert arguments into target types of parameters if possible."""
    return type_checked_call_skip_convert(func)(*args, **kwargs)


def _merge_args_and_kwargs(func: Callable[..., TypeT],
                           *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Returns one kwargs dictionary or
    raises an exeption in case of overlapping-name problems."""
    signature = inspect.signature(func)
    param_names = [param.name for param in signature.parameters.values()]
    if len(args) > len(param_names):
        raise TypeError(f'Too many parameters for {func.__name__}.')
    args_as_kwargs = dict(zip(param_names, list(args)))
    keys_in_args_and_kwargs = set.intersection(set(args_as_kwargs.keys()),
                                               set(kwargs.keys()))
    if keys_in_args_and_kwargs:
        raise TypeError(f'The following parameters are given as '
                        f'arg and kwarg in call of {func.__name__}: '
                        f'{keys_in_args_and_kwargs}')

    return {**args_as_kwargs, **kwargs}


def _unpack_dict(func: WrappedOrFunc,
                 data: Dict[str, Any],
                 skip_superfluous: bool,
                 convert_types: bool) -> Any:
    """Constructs an object in a type-safe way from a dictionary."""

    assert __is_dict(data), 'Argument data needs to be a dictionary.'

    signature = inspect.signature(__unwrap_decorator_type(func))
    ctor_params: Dict[str, Any] = {}

    if not skip_superfluous:
        param_names = [param.name for param in signature.parameters.values()]
        argument_names = data.keys()
        superfluous = set(argument_names) - set(param_names)
        if superfluous:
            raise TypeError(f'Superfluous parameters in call: {superfluous}')

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
                raise TypeError(f'Key {param.name} is missing.')
        else:
            ctor_params[param.name] = __get_value(param.annotation,
                                                  data[param.name],
                                                  param.name,
                                                  skip_superfluous,
                                                  convert_types)

    return __unwrap_decorator_type(func)(**ctor_params)


def __get_value(func: WrappedOrFunc, value: Any, log_name: str,
                skip_superfluous: bool, convert_types: bool) -> Any:
    """Convert a single value into target type if possible."""
    if __is_list(value):
        return __get_list_value(func, value, log_name,
                                skip_superfluous, convert_types)

    if __is_dict(value):
        return __get_dict_value(func, value)  # Use settings of inner value

    allowed_types = list(map(__unwrap_decorator_type, __get_union_types(func) \
        if __is_optional_type(func) \
        else [func]))

    if func is inspect.Parameter.empty and log_name != 'self':
        raise TypeError(f'Parameter {log_name} of target function '
                        'is missing a type annotation.')

    if Any not in allowed_types and log_name != 'self':
        if not __isinstanceofone(value, allowed_types):
            value_type = type(value)
            if convert_types:
                if __is_optional_type(func):
                    func = __get_optional_type(func)
                try:
                    return func(value)
                except ValueError:
                    raise TypeError(f'Can not convert {value} '
                                    f'from type {__get_type_name(value_type)} '
                                    f'into type {__get_type_name(func)} '
                                    f'for key {log_name}.')

            raise TypeError(f'Key {log_name} has incorrect type: '
                            f'{__get_type_name(value_type)} instead of '
                            f'{__get_type_name(func)}.')

    return value


def __get_list_value(func: Callable[..., TypeT], value: Any,
                     log_name: str,
                     skip_superfluous: bool, convert_types: bool) -> Any:
    if not __is_list_type(func) and \
            not __is_optional_list_type(func):
        raise TypeError(f'No list expected for {log_name}')
    result = []
    result_elem_type = __get_list_type_elem_type(func)
    for elem in value:
        result.append(__get_value(result_elem_type,
                                  elem, value,
                                  skip_superfluous, convert_types))
    return result


def __get_dict_value(func: Callable[..., TypeT], value: Any) -> Any:
    assert __is_dict(value)
    if __is_optional_type(func):
        return __get_optional_type(func)(**value)  # type: ignore
    return func(**value)


def __is_union_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a Union."""
    if VER_3_7_AND_UP:
        return (the_type is Union or
                __is_instance(the_type, _GenericAlias) and __type_origin_is(the_type, Union))
    return __is_instance(the_type, _Union)


def __is_list_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a List."""
    try:
        if VER_3_7_AND_UP:
            return __is_instance(the_type,
                                 _GenericAlias) and __type_origin_is(the_type, list)
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
            return __is_instance(the_type,
                                 _GenericAlias) and __type_origin_is(the_type, dict)
        return issubclass(the_type, Dict)  # type: ignore
    except TypeError:
        return False


def __type_origin_is(the_type: Callable[..., TypeT], origin: Any) -> bool:
    assert hasattr(the_type, '__origin__')
    return the_type.__origin__ is origin  # type: ignore


def __get_union_types(union_type: Callable[..., TypeT]) -> List[Callable[..., TypeT]]:
    """Return all types a Union can hold."""
    assert __is_union_type(union_type)
    return union_type.__args__  # type: ignore


def __get_optional_type(optional_type: Callable[..., TypeT]) -> Type[TypeT]:
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
            if __is_instance(value, the_type):  # type: ignore
                return True
        except TypeError:
            pass
    return False


def __is_optional_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is an Optional."""
    if not __is_union_type(the_type):
        return False
    union_args = __get_union_types(the_type)
    return len(union_args) == 2 and __is_instance(None, union_args[1])


def __is_dict(value: TypeT) -> bool:
    """Return True if the value is a dictionary."""
    return isinstance(value, dict)


def __is_list(value: TypeT) -> bool:
    """Return True if the value is a list."""
    return isinstance(value, list)


def __unwrap_decorator_type(func: WrappedOrFunc) -> Callable[..., Any]:
    """Get the actual type returned by the internal wrapper"""
    if _is_wrapped_func(func):
        return getattr(func, '_undictify_wrapped_func')  # type: ignore
    return func


def __is_instance(value: TypeT, the_type: Callable[..., TypeT]) -> bool:
    return isinstance(value, the_type)  # type: ignore
