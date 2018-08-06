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


class _WrappedFunction:
    def __init__(self, func: Callable[..., TypeT],
                 skip: bool, convert: bool) -> None:
        self._skip: bool = skip
        self._convert: bool = convert
        self._func: Callable[..., TypeT] = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _type_checked_apply_wrapped(self, *args, **kwargs)

    def call_wrapped(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function without any further preprocessing."""
        return self.get_wrapped_function()(*args, **kwargs)

    def get_wrapped_function(self) -> Callable[..., TypeT]:
        """Returns the actual function being called."""
        return self._func

    def get_skip(self) -> bool:
        """Is skip flag set?"""
        return self._skip

    def get_convert(self) -> bool:
        """Is convert flag set?"""
        return self._convert


WrappedOrFunc = Union[_WrappedFunction, Callable[..., TypeT]]


def __create_wrapped_function(func: WrappedOrFunc,
                              skip: bool, convert: bool) -> _WrappedFunction:
    """Return input function with new parameters if already wrapped."""
    if isinstance(func, _WrappedFunction):
        return __wrapped_function_with_new_params(func, skip, convert)
    return _WrappedFunction(func, skip, convert)


def __wrapped_function_with_new_params(wrapped_func: _WrappedFunction,
                                       skip: bool, convert: bool) -> _WrappedFunction:
    return __create_wrapped_function(wrapped_func.get_wrapped_function(), skip, convert)


def _type_checked_apply_wrapped(wrapped_func: _WrappedFunction,
                                *args: Any, **kwargs: Any) -> Any:
    """Type check the arguments of a function call."""
    return __unpack_dict(wrapped_func,
                         __merge_args_and_kwargs(wrapped_func.get_wrapped_function(),
                                                 *args, **kwargs),
                         wrapped_func.get_skip(), wrapped_func.get_convert())


def type_checked_call(func: Callable[..., TypeT]) -> _WrappedFunction:
    """Decorator that type checks arguments to every call of a function."""

    return __create_wrapped_function(func, False, False)


def type_checked_call_skip(func: Callable[..., TypeT]) -> _WrappedFunction:
    """Decorator that type checks arguments to every call of a function.
    It skips all keyword arguments that the function does not take."""

    return __create_wrapped_function(func, True, False)


def type_checked_call_convert(func: Callable[..., TypeT]) -> _WrappedFunction:
    """Decorator that type checks arguments to every call of a function.
    It converts arguments into target types of parameters if possible."""

    return __create_wrapped_function(func, False, True)


def type_checked_call_skip_convert(func: Callable[..., TypeT]) -> _WrappedFunction:
    """Decorator that type checks arguments to every call of a function.
    It skips all keyword arguments that the function does not take and
    converts arguments into target types of parameters if possible."""

    return __create_wrapped_function(func, True, True)


def type_checked_apply(func: WrappedOrFunc,
                       *args: Any, **kwargs: Any) -> Any:
    """Type check the arguments of a function call."""
    if isinstance(func, _WrappedFunction):
        __wrapped_function_with_new_params(func, False, False)(*args, **kwargs)
    return __create_wrapped_function(func, False, False)(*args, **kwargs)


def type_checked_apply_skip(func: WrappedOrFunc,
                            *args: Any, **kwargs: Any) -> Any:
    """Type check the arguments of a function call.
    Skips all keyword arguments that the function does not take."""
    if isinstance(func, _WrappedFunction):
        __wrapped_function_with_new_params(func, True, False)(*args, **kwargs)
    return __create_wrapped_function(func, True, False)(*args, **kwargs)


def type_checked_apply_convert(func: WrappedOrFunc,
                               *args: Any, **kwargs: Any) -> Any:
    """Type check the arguments of a function call.
    Convert arguments into target types of parameters if possible."""
    if isinstance(func, _WrappedFunction):
        __wrapped_function_with_new_params(func, False, True)(*args, **kwargs)
    return __create_wrapped_function(func, False, True)(*args, **kwargs)


def type_checked_apply_skip_convert(func: WrappedOrFunc,
                                    *args: Any, **kwargs: Any) -> Any:
    """Decorator that type checks arguments to every call of a function.
    Skips all keyword arguments that the function does not take and
    convert arguments into target types of parameters if possible."""
    if isinstance(func, _WrappedFunction):
        __wrapped_function_with_new_params(func, True, True)(*args, **kwargs)
    return __create_wrapped_function(func, True, True)(*args, **kwargs)


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


def __unpack_dict(func: Union[_WrappedFunction, Callable[..., TypeT]],
                  data: Dict[str, Any],
                  skip_superfluous: bool,
                  convert_types: bool) -> Any:
    """Constructs an object in a type-safe way from a dictionary."""

    assert __is_dict(data), 'Argument data needs to be a dictionary.'

    signature = inspect.signature(func.get_wrapped_function()) \
        if isinstance(func, _WrappedFunction) \
        else inspect.signature(func)
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

    return func.call_wrapped(**ctor_params) \
        if isinstance(func, _WrappedFunction) \
        else func(**ctor_params)


def __get_value(target_type: WrappedOrFunc, value: Any, log_name: str,
                skip_superfluous: bool, convert_types: bool) -> Any:
    """Convert a single value into target type if possible."""
    if __is_list(value):
        return __get_list_value(target_type, value, log_name,
                                skip_superfluous, convert_types)

    if __is_dict(value):
        return __get_dict_value(target_type, value,
                                skip_superfluous, convert_types)

    if isinstance(target_type, _WrappedFunction):
        __get_undictify_wrapped_func_value(target_type, value)

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


def __get_undictify_wrapped_func_value(target_type: _WrappedFunction,
                                       value: Any) -> Any:
    if __is_dict(value):
        return target_type(**value)
    wrapped_function = target_type.get_wrapped_function()
    if __is_instance(value, wrapped_function):
        return value
    return wrapped_function(value)


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


def __unwrap_decorator_type(the_type: WrappedOrFunc) -> Callable[..., Any]:
    """Get the actual type returned by the internal wrapper"""
    try:
        if isinstance(the_type, _WrappedFunction):
            return the_type.get_wrapped_function()
        return the_type
    except TypeError:
        return the_type


def __is_undictify_wrapped_func(the_type: Callable[..., TypeT]) -> bool:
    return __is_instance(the_type, _WrappedFunction)


def __get_undictify_wrapped_func(func: _WrappedFunction) -> Callable[..., Any]:
    assert __is_undictify_wrapped_func(func)
    return func.get_wrapped_function()


def __is_instance(value: TypeT, the_type: Callable[..., TypeT]) -> bool:
    return isinstance(value, the_type)  # type: ignore
