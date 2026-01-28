import copy
from contextlib import contextmanager

class Config:
    """Global configuration parameter manager """
    def __init__(self):
        # Default parameter values
        self._defaults = {
            'max_mpol': 21,
            'max_ntor': 21,
            'jit': True,
            'cache': True,
        }
        # Current parameter values (initialized as deep copy of defaults)
        self._current = copy.deepcopy(self._defaults)

    def __getattr__(self, name):
        """Access parameters via dot notation: config.verbose"""
        if name not in self._current:
            raise AttributeError(f"Parameter '{name}' does not exist")
        return self._current[name]

    def __setattr__(self, name, value):
        """Set parameters via dot notation: config.verbose = True"""
        if name in ('_defaults', '_current'):
            # Directly set internal attributes to avoid recursion
            super().__setattr__(name, value)
        elif name in self._current:
            self._current[name] = value
        else:
            raise AttributeError(f"Cannot set undefined parameter '{name}'")

    def reset(self):
        """Reset all parameters to default values"""
        self._current = copy.deepcopy(self._defaults)

    def update(self, params):
        """Batch update parameters using a dictionary"""
        for key, value in params.items():
            if key not in self._current:
                raise KeyError(f"Undefined parameter '{key}'")
            self._current[key] = value

    @contextmanager
    def rc_context(self, params=None):
        """Context manager for temporary parameter modification"""
        original = copy.deepcopy(self._current)
        try:
            if params:
                self.update(params)
            yield
        finally:
            self._current = original

    def add_param(self, name, default_value):
        """Dynamically register a new parameter"""
        if name in self._defaults:
            raise KeyError(f"Parameter '{name}' already exists")
        self._defaults[name] = default_value
        self._current[name] = default_value

# Global configuration instance (main entry point)
tfParams = Config()
