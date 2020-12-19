"""
undictify - Type-checked function calls at runtime
"""
import enum
import inspect
from dataclasses import InitVar, is_dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_type_hints, Tuple
from typing import _GenericAlias  # type: ignore

TypeT = TypeVar('TypeT')


class ConverterTag(enum.Enum):
    """When to apply a converter"""
    OPTIONAL = 1
    MANDATORY = 2


Converter = Callable[[Any], Any]
Converters = Dict[str, Converter]
TaggedConverter = Tuple[Callable[[Any], Any], ConverterTag]
TaggedConverters = Dict[str, TaggedConverter]
OptionalTaggedConverters = Optional[TaggedConverters]


def optional_converter(converter: Converter) -> TaggedConverter:
    """Create an optional converter"""
    return converter, ConverterTag.OPTIONAL


def mandatory_converter(converter: Converter) -> TaggedConverter:
    """Create a mandatory converter"""
    return converter, ConverterTag.MANDATORY


def type_checked_constructor(skip: bool = False,
                             convert: bool = False,
                             converters: OptionalTaggedConverters = None) \
        -> Callable[[Callable[..., TypeT]], Callable[..., TypeT]]:
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

        def wrapper(original_func: Callable[..., TypeT],
                    original_signature: inspect.Signature) \
                -> Callable[..., TypeT]:

            @wraps(original_func)
            def inner(first_arg: Any, *args: Any, **kwargs: Any) -> TypeT:
                kwargs_dict = _merge_args_and_kwargs(
                    original_signature, func_name, [first_arg] + list(args),
                    kwargs)
                return _unpack_dict(  # type: ignore
                    original_func,
                    original_signature,
                    first_arg,
                    kwargs_dict,
                    skip,
                    convert,
                    _optional_converters_to_converters(converters))

            return inner

        signature_new = _get_signature(func.__new__)
        signature_new_param_names = [param.name for param in signature_new.parameters.values()]

        if signature_new_param_names != ['args', 'kwargs']:
            func.__new__ = wrapper(func.__new__, signature_new)  # type: ignore
        else:
            func.__init__ = wrapper(func.__init__, _get_signature(func.__init__))  # type: ignore
            if is_dataclass(func) and hasattr(func, '__post_init__'):
                func.__post_init__ = wrapper(func.__post_init__,  # type: ignore
                                             _get_signature(func.__post_init__))  # type: ignore
        setattr(func, '__undictify_wrapped_func__', func)
        return func

    return call_decorator


def type_checked_call(skip: bool = False,
                      convert: bool = False,
                      converters: OptionalTaggedConverters = None) \
        -> Callable[[Callable[..., TypeT]], Callable[..., TypeT]]:
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
                convert,
                _optional_converters_to_converters(converters))

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


def _is_wrapped_func(func: WrappedOrFunc[TypeT]) -> bool:
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


def _unpack_dict(func: WrappedOrFunc[TypeT],  # pylint: disable=too-many-arguments
                 signature: inspect.Signature,
                 first_arg: Any,
                 data: Dict[str, Any],
                 skip_superfluous: bool,
                 convert_types: bool,
                 converters: TaggedConverters) -> Any:
    """Constructs an object in a type-safe way from a dictionary."""

    assert _is_dict(data), 'Argument data needs to be a dictionary.'

    call_arguments: Dict[str, Any] = {}

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
                and not _is_optional_type(param.annotation) \
                and not _is_union_of_builtins_type(param.annotation):
            raise TypeError('Union members in target function '
                            'other than Optional or just built-in types '
                            'are not supported.')
        if param.name not in data:
            if _is_optional_type(param.annotation):
                call_arguments[param.name] = None
        else:
            call_arguments[param.name] = _get_value(param.annotation,
                                                    data[param.name],
                                                    param.name,
                                                    skip_superfluous,
                                                    convert_types,
                                                    converters)

    if first_arg is not None:
        return _unwrap_decorator_type(func)(first_arg, **call_arguments)
    return _unwrap_decorator_type(func)(**call_arguments)


def _optional_converters_to_converters(converters: OptionalTaggedConverters) -> TaggedConverters:
    """Convert None to empty dictionary"""
    if not converters:
        return {}
    return converters


def _split_converters(converters: TaggedConverters) \
        -> Tuple[Converters, Converters]:
    """Partition into mandatory and optional."""
    if not converters:
        return {}, {}
    mandatory: Converters = {}
    optional: Converters = {}
    for param_name, (converter, tag) in converters.items():
        if tag == ConverterTag.MANDATORY:
            mandatory[param_name] = converter
        elif tag == ConverterTag.OPTIONAL:
            optional[param_name] = converter
        else:
            assert False
    return mandatory, optional


# pylint: disable=too-many-arguments,too-many-return-statements,too-many-branches
def _get_value(func: WrappedOrFunc[TypeT],
               value: Any, param_name: str,
               skip_superfluous: bool, convert_types: bool,
               converters: TaggedConverters) -> Any:
    """Convert a single value into target type if possible."""
    if _is_initvar_type(func):
        return value

    if _is_list(value):
        return _get_list_value(func, value, param_name,
                               skip_superfluous, convert_types,
                               converters)

    if _is_dict(value):
        # Use settings of inner value
        return _get_dict_value(func, value, skip_superfluous, convert_types,
                               converters)

    allowed_types = list(map(_unwrap_decorator_type, _get_union_types(func) \
        if _is_optional_type(func) or _is_union_type(func) \
        else [func]))

    if func is inspect.Parameter.empty and param_name != 'self':
        raise TypeError(f'Parameter {param_name} of target function '
                        'is missing a type annotation.')

    mandatory_converters, optional_converters = _split_converters(converters)

    if param_name != 'self':
        value_type = type(value)
        if mandatory_converters and param_name in mandatory_converters:
            result = mandatory_converters[param_name](value)
            if not _isinstanceofone(result, allowed_types):
                raise TypeError(f'Mandatory custom conversion for {param_name} '
                                f'yields incorrect target type: '
                                f'{_get_type_name(type(result))}')
            return result
        if Any not in allowed_types and not _isinstanceofone(value, allowed_types):
            if _is_enum_type(func):
                for entry in func:  # type: ignore
                    if value == entry.value:
                        return entry
                raise TypeError(f'Unable to instantiate {func} from {value}.')
            if optional_converters and param_name in optional_converters:
                result = optional_converters[param_name](value)
                if not _isinstanceofone(result, allowed_types):
                    raise TypeError(f'Optional custom conversion for {param_name} '
                                    f'yields incorrect target type: '
                                    f'{_get_type_name(type(result))}')
                return result
            if convert_types:
                if _is_optional_type(func):
                    func = _get_optional_type(func)
                if _is_union_type(func):
                    raise TypeError(f'The convert flag must be set to False '
                                    f'when Unions are used to avoid '
                                    f'ambiguities. '
                                    f'Thus {value} is not converted '
                                    f'from type {_get_type_name(value_type)} '
                                    f'into type {_get_type_name(func)} '
                                    f'for key {param_name}.')
                try:
                    if isinstance(value, str) and func is bool:  # type: ignore
                        return _string_to_bool(value)
                    return func(value)
                except ValueError as ex:
                    raise TypeError(f'Can not convert {value} '
                                    f'from type {_get_type_name(value_type)} '
                                    f'into type {_get_type_name(func)} '
                                    f'for key {param_name}.') from ex

            raise TypeError(f'Key {param_name} has incorrect type: '
                            f'{_get_type_name(value_type)} instead of '
                            f'{_get_type_name(func)}.')

    return value


def _string_to_bool(value: str) -> bool:
    """In accordance to configparser.ConfigParser.getboolean"""
    value_lower = value.lower()
    if value_lower in ['1', 'yes', 'true', 'on']:
        return True
    if value_lower in ['0', 'no', 'false', 'off']:
        return False
    raise TypeError(f'Cannot convert string "{value}" to bool.')


def _get_list_value(func: Callable[..., TypeT],  # pylint: disable=too-many-arguments
                    value: Any, log_name: str,
                    skip_superfluous: bool, convert_types: bool,
                    converters: TaggedConverters) -> Any:
    if not _is_list_type(func) and \
            not _is_optional_list_type(func):
        raise TypeError(f'No list expected for {log_name}')
    result = []
    result_elem_type = _get_list_type_elem_type(func)
    for elem in value:
        result.append(_get_value(result_elem_type,
                                 elem, value,
                                 skip_superfluous, convert_types,
                                 converters))
    return result


def _get_dict_value(func: Callable[..., TypeT], value: Any,
                    skip_superfluous: bool, convert_types: bool,
                    converters: TaggedConverters) -> Any:
    assert _is_dict(value)
    if _is_optional_type(func):
        return _get_optional_type(func)(**value)  # type: ignore
    if _is_dict_type(func):
        key_type = _get_dict_key_type(func)
        value_type = _get_dict_value_type(func)
        typed_dict = {}
        for dict_key, dict_value in value.items():
            typed_dict[_get_value(key_type, dict_key, 'dict_key',
                                  skip_superfluous, convert_types,
                                  converters)] = \
                _get_value(value_type, dict_value, 'dict_value',
                           skip_superfluous, convert_types,
                           converters)
        return typed_dict
    if func is Any:
        return value
    return func(**value)


def _is_initvar_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is an InitVar

    In Python 3.7, InitVar is essentially a singleton.
        InitVar[str] == InitVar[int]
    In Python 3.8, InitVar can be a singleton, or an instance.
        InitVar[str] != InitVar[int], but InitVar == InitVar

    Therefore, the code below checks for both cases to support 3.7 and 3.8
    """
    return the_type == InitVar or isinstance(the_type, InitVar)  # type: ignore


def _is_union_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a Union."""
    return (the_type is Union or  # type: ignore
            _is_instance(the_type, _GenericAlias) and _type_origin_is(the_type, Union))


def _is_list_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a List."""
    try:
        return _is_instance(the_type,
                            _GenericAlias) and _type_origin_is(the_type, list)
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
        return _is_instance(the_type,
                            _GenericAlias) and _type_origin_is(the_type, dict)
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


def _get_dict_key_type(dict_type: Callable[..., TypeT]) -> Type[TypeT]:
    """Return the type the keys of a Dict should have."""
    assert _is_dict_type(dict_type)
    args = dict_type.__args__  # type: ignore
    assert len(args) == 2
    return args[0]  # type: ignore


def _get_dict_value_type(dict_type: Callable[..., TypeT]) -> Type[TypeT]:
    """Return the type the values of a Dict should have."""
    assert _is_dict_type(dict_type)
    args = dict_type.__args__  # type: ignore
    assert len(args) == 2
    return args[1]  # type: ignore


def _get_type_name(the_type: Callable[..., TypeT]) -> str:
    """Return a printable name of a type."""
    if _is_optional_type(the_type):
        return f'Optional[{str(_get_optional_type(the_type).__name__)}]'
    if _is_union_type(the_type):
        union_type_names = [t.__name__ for t in _get_union_types(the_type)]
        return f'Union[{", ".join(union_type_names)}]'
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
            # pylint: disable=unidiomatic-typecheck
            if type(value) == the_type:  # type: ignore
                return True
            # pylint: enable=unidiomatic-typecheck
        except TypeError:
            pass
    return False


def _is_optional_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is an Optional."""
    if not _is_union_type(the_type):
        return False
    union_args = _get_union_types(the_type)
    return any(_is_none_type(union_arg) for union_arg in union_args)


def _is_enum_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is an Enum."""
    try:
        return issubclass(the_type, enum.Enum)  # type: ignore
    except TypeError:
        return False


def _is_union_of_builtins_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is an Union only made of
    None, str, int, float and bool."""
    if not _is_union_type(the_type):
        return False
    union_args = _get_union_types(the_type)
    return all(map(_is_builtin_type, union_args))


def _is_builtin_type(the_type: Callable[..., TypeT]) -> bool:
    """Return True if the type is a NoneType, str, int, float or bool."""
    return the_type in [str, int, bool, float, type(None)]  # type: ignore


def _is_none_type(value: TypeT) -> bool:
    """Return True if the value is of NoneType."""
    return value is type(None)


def _is_dict(value: TypeT) -> bool:
    """Return True if the value is a dictionary."""
    return isinstance(value, dict)


def _is_list(value: TypeT) -> bool:
    """Return True if the value is a list."""
    return isinstance(value, list)


def _unwrap_decorator_type(func: WrappedOrFunc[TypeT]) -> Callable[..., Any]:
    """Get the actual type returned by the internal wrapper"""
    if _is_wrapped_func(func):
        return getattr(func, '__undictify_wrapped_func__')  # type: ignore
    return func


def _is_instance(value: TypeT, the_type: Callable[..., TypeT]) -> bool:
    return isinstance(value, the_type)  # type: ignore


def _get_signature(func: WrappedOrFunc[TypeT]) -> inspect.Signature:
    if hasattr(func, '__annotations__'):
        # https://stackoverflow.com/questions/53450624/hasattr-telling-lies-attributeerror-method-object-has-no-attribute-annot
        try:
            func.__annotations__ = get_type_hints(func)
            return inspect.signature(func)
        except AttributeError:
            return inspect.signature(func)
    return inspect.signature(func)
