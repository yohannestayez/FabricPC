"""
Unified registry system for FabricPC.

This module provides a generic Registry class that handles:
- Class registration with validation
- Type lookup
- Entry point discovery for external packages

Used by node registry, energy registry, and activation registry.
"""

from typing import Type, Dict, Any, List, Callable, Optional
import sys
import warnings


class RegistrationError(Exception):
    """Base exception for registration failures."""

    pass


class Registry:
    """
    Generic registry for registering classes with validation.

    This class provides a unified pattern for:
    - Decorator-based registration
    - Class validation (required attributes, methods)
    - Case-insensitive type lookup
    - Entry point discovery for external packages

    Example usage:
        # Create registry instance
        node_registry = Registry(
            name="node",
            entry_point_group="fabricpc.nodes",
            required_attrs=["CONFIG_SCHEMA", "DEFAULT_ENERGY_CONFIG"],
            required_methods=["get_slots", "initialize_params", "forward"]
        )

        # Register a class
        @node_registry.register("linear")
        class LinearNode:
            CONFIG_SCHEMA = {}
            ...

        # Look up a class
        cls = node_registry.get("linear")
    """

    def __init__(
        self,
        name: str,
        entry_point_group: str,
        required_attrs: List[str] = None,
        required_methods: List[str] = None,
        attr_validators: Dict[str, Callable[[Any, str], None]] = None,
    ):
        """
        Initialize a registry.

        Args:
            name: Human-readable name for this registry (e.g., "node", "energy")
            entry_point_group: Entry point group name for external discovery
            required_attrs: List of required class attributes
            required_methods: List of required methods (must not be abstract)
            attr_validators: Dict of attribute name -> validation function
                            Validation function takes (attr_value, type_name) and raises on error
        """
        self.name = name
        self.entry_point_group = entry_point_group
        self.required_attrs = required_attrs or []
        self.required_methods = required_methods or []
        self.attr_validators = attr_validators or {}
        self._registry: Dict[str, Type] = {}
        self._error_class = RegistrationError

    def set_error_class(self, error_class: Type[Exception]) -> None:
        """Set custom error class for this registry."""
        self._error_class = error_class

    def register(self, type_name: str):
        """
        Decorator to register a class with the registry.

        Args:
            type_name: Unique identifier for this type (case-insensitive)

        Returns:
            Decorator function

        Raises:
            RegistrationError: If registration fails (duplicate, missing attrs/methods)
        """

        def decorator(cls: Type) -> Type:
            type_lower = type_name.lower()

            # Check for duplicate registration
            if type_lower in self._registry:
                existing = self._registry[type_lower]
                if existing is not cls:
                    raise self._error_class(
                        f"{self.name.title()} type '{type_name}' already registered by {existing.__name__}"
                    )
                # Same class registered twice - silently allow (import idempotency)
                return cls

            # Validate class
            self._validate_class(cls, type_name)

            # Register
            self._registry[type_lower] = cls

            # Store the registered name on the class for introspection
            cls._registered_type = type_lower

            return cls

        return decorator

    def _validate_class(self, cls: Type, type_name: str) -> None:
        """
        Validate that a class implements the required interface.

        Args:
            cls: The class to validate
            type_name: The type name being registered (for error messages)

        Raises:
            RegistrationError: If validation fails
        """
        # Check required attributes
        for attr_name in self.required_attrs:
            if not hasattr(cls, attr_name):
                raise self._error_class(
                    f"{self.name.title()} type '{type_name}': missing required {attr_name} attribute."
                )

            # Run custom validator if present
            if attr_name in self.attr_validators:
                attr_value = getattr(cls, attr_name)
                self.attr_validators[attr_name](
                    attr_value, type_name, self._error_class
                )

        # Check required methods
        for method_name in self.required_methods:
            method = getattr(cls, method_name, None)
            if method is None:
                raise self._error_class(
                    f"{self.name.title()} type '{type_name}': missing required method '{method_name}'"
                )
            # Check it's not still abstract
            if getattr(method, "__isabstractmethod__", False):
                raise self._error_class(
                    f"{self.name.title()} type '{type_name}': method '{method_name}' is abstract"
                )

    def get(self, type_name: str) -> Type:
        """
        Get a registered class by its type name.

        Args:
            type_name: The registered type (case-insensitive)

        Returns:
            The registered class

        Raises:
            ValueError: If type is not registered
        """
        type_lower = type_name.lower()
        if type_lower not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Unknown {self.name} type '{type_name}'. "
                f"Available types: {available}"
            )
        return self._registry[type_lower]

    def list_types(self) -> List[str]:
        """Return list of all registered types."""
        return sorted(self._registry.keys())

    def unregister(self, type_name: str) -> None:
        """
        Remove a type from the registry.
        Primarily for testing purposes.

        Args:
            type_name: The type to unregister (case-insensitive)
        """
        type_lower = type_name.lower()
        if type_lower in self._registry:
            del self._registry[type_lower]

    def clear(self) -> None:
        """Clear all registrations. For testing only."""
        self._registry.clear()

    def discover_external(self) -> None:
        """
        Discover and register classes from installed packages via entry points.

        Looks for packages with entry points in the configured group.
        Each entry point should map a type name to a class.
        """
        try:
            if sys.version_info >= (3, 10):
                from importlib.metadata import entry_points

                eps = entry_points(group=self.entry_point_group)
            else:
                from importlib.metadata import entry_points

                all_eps = entry_points()
                eps = all_eps.get(self.entry_point_group, [])

            for ep in eps:
                try:
                    cls = ep.load()
                    type_name = ep.name.lower()

                    # Skip if already registered (built-in takes precedence)
                    if type_name in self._registry:
                        continue

                    # Validate and register
                    self._validate_class(cls, type_name)
                    self._registry[type_name] = cls
                    cls._registered_type = type_name

                except Exception as e:
                    warnings.warn(
                        f"Failed to load {self.name} '{ep.name}' from {ep.value}: {e}",
                        RuntimeWarning,
                    )
        except Exception as e:
            # Entry point discovery failed entirely - not critical
            warnings.warn(
                f"{self.name.title()} entry point discovery failed: {e}", RuntimeWarning
            )

    def __contains__(self, type_name: str) -> bool:
        """Check if a type is registered."""
        return type_name.lower() in self._registry

    def __len__(self) -> int:
        """Return number of registered types."""
        return len(self._registry)


# =============================================================================
# Attribute Validators
# =============================================================================


def validate_config_schema(
    attr_value: Any, type_name: str, error_class: Type[Exception]
) -> None:
    """Validate CONFIG_SCHEMA attribute is a dict."""
    if not isinstance(attr_value, dict):
        raise error_class(
            f"type '{type_name}': CONFIG_SCHEMA must be a dict, "
            f"got {type(attr_value).__name__}"
        )


def validate_default_energy_config(
    attr_value: Any, type_name: str, error_class: Type[Exception]
) -> None:
    """Validate DEFAULT_ENERGY_CONFIG attribute."""
    if not isinstance(attr_value, dict):
        raise error_class(
            f"type '{type_name}': DEFAULT_ENERGY_CONFIG must be a dict, "
            f"got {type(attr_value).__name__}"
        )
    if "type" not in attr_value:
        raise error_class(
            f"type '{type_name}': DEFAULT_ENERGY_CONFIG must have a 'type' key"
        )
