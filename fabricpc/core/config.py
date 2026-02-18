"""
Unified configuration validation for FabricPC.

This module provides:
- ConfigValidationError for config-related errors
- transform_shorthand() to convert string configs to dict form
- validate_config() for schema-based validation with defaults

All configurable objects (nodes, energy, activations) use this system.
"""

from typing import Dict, Any, Union, Tuple, Type


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


def transform_shorthand(
    config: Union[str, Dict[str, Any]], context: str = ""
) -> Dict[str, Any]:
    """
    Transform string config shorthand to dict form.

    Rules:
    - String config: "sigmoid" -> {"type": "sigmoid"}
    - Dict config: Must contain "type" field

    Args:
        config: String shorthand or dict config
        context: Context string for error messages (e.g., "activation", "energy")

    Returns:
        Dict config with "type" field

    Raises:
        ConfigValidationError: If dict config missing "type" or invalid type

    Examples:
        >>> transform_shorthand("sigmoid")
        {"type": "sigmoid"}

        >>> transform_shorthand({"type": "leaky_relu", "alpha": 0.1})
        {"type": "leaky_relu", "alpha": 0.1}

        >>> transform_shorthand({"alpha": 0.1})  # Raises error
        ConfigValidationError: ...
    """
    context_prefix = f"{context}: " if context else ""

    if isinstance(config, str):
        return {"type": config}

    if isinstance(config, dict):
        if "type" not in config:
            raise ConfigValidationError(
                f"{context_prefix}dict config must include 'type' field. "
                f"Got keys: {list(config.keys())}"
            )
        return config

    raise ConfigValidationError(
        f"{context_prefix}config must be str or dict, got {type(config).__name__}"
    )


def validate_config(
    schema: Dict[str, Dict[str, Any]], config: Dict[str, Any], context: str = ""
) -> Dict[str, Any]:
    """
    Validate config against schema and apply defaults.

    Schema format:
        {
            "param_name": {
                "type": type_or_tuple,     # Python type(s) for validation
                "default": value,           # Default value (optional if present)
                "required": bool,           # True if no default
                "choices": [values],        # Allowed values (optional)
                "description": "...",       # Human-readable description
            }
        }

    Args:
        schema: Schema dict defining expected fields
        config: User-provided config dict
        context: Context string for error messages

    Returns:
        Validated config dict with defaults applied

    Raises:
        ConfigValidationError: If validation fails

    Examples:
        >>> schema = {
        ...     "alpha": {"type": float, "default": 0.01},
        ...     "mode": {"type": str, "required": True}
        ... }
        >>> validate_config(schema, {"mode": "train"})
        {"mode": "train", "alpha": 0.01}
    """
    if schema is None:
        return config if config else {}

    context_prefix = f"{context}: " if context else ""
    result = dict(config) if config else {}

    for field_name, field_spec in schema.items():
        if field_name in result:
            value = result[field_name]

            # Type validation
            expected_type = field_spec.get("type")
            if expected_type is not None:
                if not isinstance(value, expected_type):
                    # Format type name for error message
                    if isinstance(expected_type, tuple):
                        type_names = " or ".join(t.__name__ for t in expected_type)
                    else:
                        type_names = expected_type.__name__

                    raise ConfigValidationError(
                        f"{context_prefix}'{field_name}' must be {type_names}, "
                        f"got {type(value).__name__}"
                    )

            # Choice validation
            choices = field_spec.get("choices")
            if choices is not None and value not in choices:
                raise ConfigValidationError(
                    f"{context_prefix}'{field_name}' must be one of {choices}, "
                    f"got '{value}'"
                )

        elif field_spec.get("required", False):
            # Missing required field
            description = field_spec.get("description", "no description")
            raise ConfigValidationError(
                f"{context_prefix}missing required field '{field_name}'. "
                f"Description: {description}"
            )

        elif "default" in field_spec:
            # Apply default
            default = field_spec["default"]
            # Deep copy dicts and lists to avoid shared mutable state
            if isinstance(default, dict):
                result[field_name] = dict(default)
            elif isinstance(default, list):
                result[field_name] = list(default)
            else:
                result[field_name] = default

    return result


def validate_typed_config(
    config: Union[str, Dict[str, Any]], get_class_fn, context: str = ""
) -> Dict[str, Any]:
    """
    Validate a typed config (one that has a "type" field referencing a registry).

    This combines transform_shorthand with schema validation by:
    1. Transforming string shorthand to dict
    2. Looking up the class from the type
    3. Validating against the class's CONFIG_SCHEMA

    Args:
        config: String shorthand or dict config with "type" field
        get_class_fn: Function to get class from type name (e.g., get_energy_class)
        context: Context string for error messages

    Returns:
        Validated config dict with defaults applied

    Raises:
        ConfigValidationError: If validation fails
    """
    context_prefix = f"{context}: " if context else ""

    # Transform shorthand
    config_dict = transform_shorthand(config, context)

    # Get type and look up class
    type_name = config_dict.get("type")
    if type_name is None:
        raise ConfigValidationError(f"{context_prefix}missing required 'type' field")

    try:
        cls = get_class_fn(type_name)
    except ValueError as e:
        raise ConfigValidationError(f"{context_prefix}{e}") from e

    # Validate against class schema
    schema = getattr(cls, "CONFIG_SCHEMA", None)
    return validate_config(schema, config_dict, context)
