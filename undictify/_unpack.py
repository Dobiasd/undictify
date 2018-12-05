"""
undictify - Type-checked function calls at runtime
"""
import inspect
import sys
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, get_type_hints

VER_3_7_AND_UP = sys.version_info[:3] >= (3, 7, 0)  # PEP 560

# pylint: disable=no-name-in-module
if VER_3_7_AND_UP:
    from typing import _GenericAlias  # type: ignore
else:
    from typing import _Union  # type: ignore
# pylint: enable=no-name-in-module

TypeT = TypeVar('TypeT')


def type_checked_constructor(skip: bool = False,
                             convert: bool = False) -> Callable[[Callable[..., TypeT]],
                                                                Callable[..., TypeT]]:
    """Replaces the constructor of the given class (in-place)
    with type-checked calls."""

    def call_decorator(func: Callable[..., TypeT]) -> Callable[..., TypeT]:
        if not inspect.isclass(func):
            raise TypeError('@_type_checked_constructor may only be used for classes.')

        if _is_wrapped_func(func):
            raise TypeError('Class is already wrapped by undictify.')

        # Ideally we could prevent type_checked_constructor to be used
        # as a normal function instead of a decorator.
        # However this turns out to be very tricky,
        # and given solutions break down on some corner cases.
        # https://stackoverflow.com/questions/52191968/check-if-a-function-was-called-as-a-decorator

        func_name = _get_log_name(func)

        signature_new = _get_signature(func.__new__)
        signature_new_param_names = [param.name for param in signature_new.parameters.values()]
        if signature_new_param_names != ['args', 'kwargs']:
            signature_ctor = signature_new
            replace_init = False
            original_ctor = func.__new__
        else:
            original_ctor = func.__init__  # type: ignore
            signature_ctor = _get_signature(original_ctor)
            replace_init = True

        @wraps(original_ctor)
        def wrapper(first_arg: Any, *args: Any, **kwargs: Any) -> TypeT:
            kwargs_dict = _merge_args_and_kwargs(
                signature_ctor, func_name, [first_arg] + list(args),
                kwargs)
            return _unpack_dict(  # type: ignore
                original_ctor,
                signature_ctor,
                first_arg,
                kwargs_dict,
                skip,
                convert)

        if replace_init:
            func.__init__ = wrapper  # type: ignore
        else:
            func.__new__ = wrapper  # type: ignore
        setattr(func, '__undictify_wrapped_func__', func)
        return func

    return call_decorator


def type_checked_call(skip: bool = False,
                      convert: bool = False) -> Callable[[Callable[..., TypeT]],
                                                         Callable[..., TypeT]]:
    """Wrap function with type checks."""

    def call_decorator(func: Callable[..., TypeT]) -> Callable[..., TypeT]:
        if _is_wrapped_func(func):
            raise TypeError('Function is already wrapped by undictify.')

        signature = _get_signature(func)
        func_name = _get_log_name(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> TypeT:
            kwargs_dict = _merge_args_and_kwargs(signature, func_name,
                                                 args, kwargs)
            return _unpack_dict(  # type: ignore
                func,
                signature,
                None,
                kwargs_dict,
                skip,
                convert)

        setattr(wrapper, '__undictify_wrapped_func__', func)
        return wrapper

    return call_decorator


def _get_log_name(var: Any) -> str:
    """Return var.__name__ if available, 'this object' otherwise."""
    try:
        return str(var.__name__)
    except AttributeError:
        return 'this object'


WrappedOrFunc = Callable[..., TypeT]


def _is_wrapped_func(func: WrappedOrFunc) -> bool:
    return hasattr(func, '__undictify_wrapped_func__')


def _merge_args_and_kwargs(signature: inspect.Signature, name: str,
                           args: Any, kwargs: Any) -> Dict[str, Any]:
    """Returns one kwargs dictionary or
    raises an exeption in case of overlapping-name problems."""
    param_names = [param.name for param in signature.parameters.values()]
    if len(args) > len(param_names):
        raise TypeError(f'Too many parameters for {name}.')
    args_as_kwargs = dict(zip(param_names, list(args)))
    keys_in_args_and_kwargs = set.intersection(set(args_as_kwargs.keys()),
                                               set(kwargs.keys()))
    if keys_in_args_and_kwargs:
        raise TypeError(f'The following parameters are given as '
                        f'arg and kwarg in call of {name}: '
                        f'{keys_in_args_and_kwargs}')

    return {**args_as_kwargs, **kwargs}


def _unpack_dict(func: WrappedOrFunc,  # pylint: disable=too-many-arguments
                 signature: inspect.Signature,
                 first_arg: Any,
                 data: Dict[str, Any],
                 skip_superfluous: bool,
                 convert_types: bool) -> Any:
    """Constructs an object in a type-safe way from a dictionary."""

    assert _is_dict(data), 'Argument data needs to be a dictionary.'

    ctor_params: Dict[str, Any] = {}

    if not skip_superfluous:
        param_names = [param.name for param in signature.parameters.values()]
        argument_names = data.keys()
        superfluous = set(argument_names) - set(param_names)
        if superfluous:
            raise TypeError(f'Superfluous parameters in call: {superfluous}')

    parameter_values = list(signature.parameters.values())
    if first_arg is not None:
        parameter_values = parameter_values[1:]
    for param in parameter_values:
        if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise TypeError('Only parameters of kind POSITIONAL_OR_KEYWORD '
                            'supported in target functions.')
        if _is_union_type(param.annotation) \
                and not _is_optional_type(param.annotation):
            raise TypeError('Union members in target function other than Optional '
                            'are not supported.')
        if _is_dict_type(param.annotation):
            raise TypeError('Dict members in target function are not supported.')
        if param.name not in data:
            if _is_optional_type(param.annotation):
                ctor_params[param.name] = None
        else:
            ctor_params[param.name] = _get_value(param.annotation,
                                                 data[param.name],
                                                 param.name,
                                                 skip_superfluous,
                                                 convert_types)

    if first_arg is not None:
        return _unwrap_decorator_type(func)(first_arg, **ctor_params)
    return _unwrap_decorator_type(func)(**ctor_params)


def _get_value(func: WrappedOrFunc, value: Any, log_name: str,
               skip_superfluous: bool, convert_types: bool) -> Any:
    """Convert a single value into target type if possible."""
    if _is_list(value):
        return _get_list_value(func, value, log_name,
                               skip_superfluous, convert_types)

    if _is_dict(value):
        return _get_dict_value(func, value)  # Use settings of inner value

    allowed_types = list(map(_unwrap_decorator_type, _get_union_types(func) \
        if _is_optional_type(func) \
        else [func]))

    if func is inspect.Parameter.empty and log_name != 'self':
        raise TypeError(f'Parameter {log_name} of target function '
                        'is missing a type annotation.')

    if Any not in allowed_types and log_name != 'self':
        if not _isinstanceofone(value, allowed_types):
            value_type = type(value)
            if convert_types:
                if _is_optional_type(func):
                    func = _get_optional_type(func)
                try:
                    return func(value)
                except ValueError:
                    raise TypeError(f'Can not convert {value} '
                                    f'from type {_get_type_name(value_type)} '
                                    f'into type {_get_type_name(func)} '
                                    f'for key {log_name}.')

            raise TypeError(f'Key {log_name} has incorrect type: '
                            f'{_get_type_name(value_type)} instead of '
                            f'{_get_type_name(func)}.')

    return value


def _get_list_value(func: Callable[..., TypeT], value: Any,
                    log_name: str,
                    skip_superfluous: bool, convert_types: bool) -> Any:
    if not _is_list_type(func) and \
            not _is_optional_list_type(func):
        raise TypeError(f'No list expected for {log_name}')
    result = []
    result_elem_type = _get_list_type_elem_type(func)
    for elem in value:
        result.append(_get_value(result_elem_type,
                                 elem, value,
                                 skip_superfluous, convert_types))
    return result


def _get_dict_value(func: Callable[..., TypeT], value: Any) -> Any:
    assert _is_dict(value)
    if _is_optional_type(func):
        return _get_optional_type(func)(**value)  # type: ignore
    return func(**value)


def _is_union_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a Union."""
    if VER_3_7_AND_UP:
        return (the_type is Union or
                _is_instance(the_type, _GenericAlias) and _type_origin_is(the_type, Union))
    return _is_instance(the_type, _Union)


def _is_list_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a List."""
    try:
        if VER_3_7_AND_UP:
            return _is_instance(the_type,
                                _GenericAlias) and _type_origin_is(the_type, list)
        return issubclass(the_type, List)  # type: ignore
    except TypeError:
        return False


def _is_optional_list_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a Optional[List]."""
    if _is_list_type(the_type):
        return True
    if _is_optional_type(the_type) and \
            _is_list_type(_get_optional_type(the_type)):
        return True
    return False


def _is_dict_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a Dict."""
    try:
        if VER_3_7_AND_UP:
            return _is_instance(the_type,
                                _GenericAlias) and _type_origin_is(the_type, dict)
        return issubclass(the_type, Dict)  # type: ignore
    except TypeError:
        return False


def _type_origin_is(the_type: Callable[..., TypeT], origin: Any) -> bool:
    assert hasattr(the_type, '__origin__')
    return the_type.__origin__ is origin  # type: ignore


def _get_union_types(union_type: Callable[..., TypeT]) -> List[Callable[..., TypeT]]:
    """Return all types a Union can hold."""
    assert _is_union_type(union_type)
    return union_type.__args__  # type: ignore


def _get_optional_type(optional_type: Callable[..., TypeT]) -> Type[TypeT]:
    """Return the type an Optional can hold."""
    assert _is_optional_type(optional_type)
    args = optional_type.__args__  # type: ignore
    assert len(args) == 2
    return args[0]  # type: ignore


def _get_type_name(the_type: Callable[..., TypeT]) -> str:
    """Return a printable name of a type."""
    if _is_optional_type(the_type):
        return f'Optional[{str(_get_optional_type(the_type).__name__)}]'
    if _is_list_type(the_type):
        return f'List[{str(_get_list_type_elem_type(the_type).__name__)}]'
    return the_type.__name__


def _get_list_type_elem_type(list_type: Callable[..., TypeT]) -> Callable[..., Any]:
    """Return the type of a single element of the list type."""
    if _is_optional_type(list_type):
        list_type = _get_optional_type(list_type)
    assert _is_list_type(list_type)
    list_args = list_type.__args__  # type: ignore
    assert len(list_args) == 1
    return list_args[0]  # type: ignore


def _isinstanceofone(value: Callable[..., TypeT], types: List[Callable[..., TypeT]]) -> bool:
    """Check if value is an instance of one of the given types."""
    for the_type in types:
        if _is_union_type(the_type):
            if _isinstanceofone(value, _get_union_types(the_type)):
                return True
        try:
            if _is_instance(value, the_type):  # type: ignore
                return True
        except TypeError:
            pass
    return False


def _is_optional_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is an Optional."""
    if not _is_union_type(the_type):
        return False
    union_args = _get_union_types(the_type)
    return len(union_args) == 2 and _is_instance(None, union_args[1])


def _is_dict(value: TypeT) -> bool:
    """Return True if the value is a dictionary."""
    return isinstance(value, dict)


def _is_list(value: TypeT) -> bool:
    """Return True if the value is a list."""
    return isinstance(value, list)


def _unwrap_decorator_type(func: WrappedOrFunc) -> Callable[..., Any]:
    """Get the actual type returned by the internal wrapper"""
    if _is_wrapped_func(func):
        return getattr(func, '__undictify_wrapped_func__')  # type: ignore
    return func


def _is_instance(value: TypeT, the_type: Callable[..., TypeT]) -> bool:
    return isinstance(value, the_type)  # type: ignore


def _get_signature(func: WrappedOrFunc) -> inspect.Signature:
    if hasattr(func, '__annotations__'):
        # https://stackoverflow.com/questions/53450624/hasattr-telling-lies-attributeerror-method-object-has-no-attribute-annot
        try:
            func.__annotations__ = get_type_hints(func)
            return inspect.signature(func)
        except AttributeError:
            return inspect.signature(func)
    return inspect.signature(func)
