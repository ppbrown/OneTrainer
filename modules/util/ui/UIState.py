# ui_state.py
import contextlib
from collections.abc import Callable
from enum import Enum
from typing import Any

from PySide6.QtCore import QObject, Signal

from modules.util.config.BaseConfig import BaseConfig


class _VarBase(QObject):
    """
    Base class for our 'Var' objects. 
    Each Var stores a value and emits 'valueChanged' when updated.
    """
    valueChanged = Signal(object)  # emits the new value

    def __init__(self, initial_value=None, nullable=False, parent=None):
        super().__init__(parent)
        self._value = initial_value
        self._nullable = nullable

        # We'll store callback IDs that the user can remove if needed
        self._trace_id_counter = 0
        self._trace_callbacks = {}

    def get(self):
        return self._value

    def set(self, new_value):
        self._value = new_value
        self.valueChanged.emit(new_value)
        # also call any local callbacks we are tracking
        for cb in self._trace_callbacks.values():
            cb()

    def trace_add(self, callback: Callable[[Any, Any, Any], None]):
        """
        Mimic tk's trace_add. We'll store the callback and call it 
        every time 'set()' is called.
        """
        self._trace_id_counter += 1
        this_id = self._trace_id_counter
        self._trace_callbacks[this_id] = callback
        return this_id

    def trace_remove(self, trace_id: int):
        self._trace_callbacks.pop(trace_id, None)


class _StringVar(_VarBase):
    """
    For storing string data. 
    We handle 'None' and empty-string logic at a higher level,
    or you can do it here if you prefer.
    """
    pass


class _BoolVar(_VarBase):
    """
    For storing bool data.
    """
    pass


class _EnumVar(_VarBase):
    """
    For storing an Enum. We'll store its string representation or 
    the enum object itself. 
    """
    def __init__(self, enum_class, initial_value=None, nullable=False, parent=None):
        super().__init__(initial_value, nullable, parent)
        self.enum_class = enum_class

    def set_enum(self, enum_value_str):
        if (enum_value_str == "" or enum_value_str == "None") and self._nullable:
            self.set(None)
        else:
            self.set(self.enum_class[enum_value_str])


class UIState:
    """
    A PySide6-based rewrite of your original UIState class that used tkinter variable objects.
    Now we use our own "Var" classes with signals/callbacks.
    """
    def __init__(self, parent, obj):
        """
        :param parent: Typically a QWidget or QMainWindow in Qt
        :param obj: The configuration object (dict or BaseConfig)
        """
        self.parent = parent
        self.obj = obj

        # Replaces __vars from tkinter code
        self.__vars: dict[str, _VarBase | 'UIState'] = self.__create_vars(obj)

        # Instead of storing command callbacks in self.__var_traces[name],
        # we store them inside each Var. But to preserve your original structure,
        # we keep a dictionary of {varname -> {trace_id -> callback}} as well.
        self.__var_traces: dict[str, dict[int, Callable[[], None]]] = {
            name: {} for name in self.__vars
        }

        self.__latest_var_trace_id = 0

    def update(self, obj):
        """
        Replaces your old method that updates the underlying object 
        and re-syncs the variable objects' values.
        """
        self.obj = obj
        self.__set_vars(obj)

    def get_var(self, name):
        """
        Splits on '.' and descends, same as in your code.
        If it finds a 'UIState', continues the search.
        If it finds a 'Var', returns it directly.
        """
        split_name = name.split('.')
        state_or_var = self

        for name_part in split_name:
            # If it's a UIState subobject, descend
            if isinstance(state_or_var, UIState):
                state_or_var = state_or_var.__vars[name_part]
            else:
                # If it's a Var (e.g. _VarBase), we can't descend further
                raise AttributeError(
                    f"Cannot descend into var '{name_part}'; no sub-state. Full path: {name}"
                )
        return state_or_var

    def add_var_trace(self, name, command: Callable[[], None]) -> int:
        """
        Mimic the original code that stored command callbacks by ID.
        We'll connect the callback to the 'Var' object as well.
        """
        self.__latest_var_trace_id += 1
        trace_id = self.__latest_var_trace_id

        var_object = self.__vars.get(name, None)
        if var_object is None:
            raise KeyError(f"No var named '{name}' in UIState.")

        # store it
        self.__var_traces[name][trace_id] = command

        # also connect to the var's signal
        # We define a small wrapper that calls your "command()" with no args
        def wrapper(_):
            # The original code calls your command with no arguments
            command()

        var_object.valueChanged.connect(wrapper)
        return trace_id

    def remove_var_trace(self, name, trace_id):
        """
        Remove the stored callback from the local dict, 
        but we'd also need to disconnect it from the Var's signal 
        or keep track of the signal connection object. 
        For simplicity, let's just remove from the dict. 
        In a real approach, we might store the actual "connection" object 
        (the returned QMetaObject.Connection) so we can disconnect it precisely.
        """
        if name in self.__var_traces:
            self.__var_traces[name].pop(trace_id, None)
        # not fully removing from signals, though.

    # -----------------------------------------------------------------------
    # "Private" methods from original code
    # -----------------------------------------------------------------------
    def __call_var_traces(self, name):
        # not used much now, we do it through signals. But let's keep it 
        # for any direct calls your code might do
        for cb in self.__var_traces[name].values():
            cb()

    def __set_str_var(self, obj, is_dict, name, var: _StringVar, nullable):
        """
        The "update" callback. This was originally a nested function that got called 
        when the tk.StringVar changed. We replicate the logic:
        If empty and nullable => set None, else set the string.
        """
        def update():
            string_var = var.get()
            if (string_var == "" or string_var == "None") and nullable:
                final_value = None
            else:
                final_value = string_var

            if is_dict:
                obj[name] = final_value
            else:
                setattr(obj, name, final_value)

            self.__call_var_traces(name)

        return update

    def __set_enum_var(self, obj, is_dict, name, var: _EnumVar, var_type, nullable):
        def update():
            string_var = var.get()
            if (string_var == "" or string_var == "None") and nullable:
                final_value = None
            else:
                # var_type is the enum class. We can do var_type[string_var]
                # but be sure to handle the case that var might store the enum object directly
                if isinstance(string_var, var_type):
                    final_value = string_var
                else:
                    final_value = var_type[string_var]

            if is_dict:
                obj[name] = final_value
            else:
                setattr(obj, name, final_value)

            self.__call_var_traces(name)

        return update

    def __set_bool_var(self, obj, is_dict, name, var: _BoolVar):
        def update():
            bool_val = var.get() or False
            if is_dict:
                obj[name] = bool_val
            else:
                setattr(obj, name, bool_val)
            self.__call_var_traces(name)

        return update

    def __set_int_var(self, obj, is_dict, name, var: _StringVar, nullable):
        def update():
            string_var = var.get()
            final_value = None
            if (string_var == "" or string_var == "None") and nullable:
                final_value = None
            elif string_var == "inf":
                final_value = int(float("inf"))
            elif string_var == "-inf":
                final_value = int(float("-inf"))
            else:
                with contextlib.suppress(ValueError):
                    final_value = int(string_var)

            if is_dict:
                obj[name] = final_value
            else:
                setattr(obj, name, final_value)

            self.__call_var_traces(name)

        return update

    def __set_float_var(self, obj, is_dict, name, var: _StringVar, nullable):
        def update():
            string_var = var.get()
            final_value = None
            if (string_var == "" or string_var == "None") and nullable:
                final_value = None
            elif string_var == "inf":
                final_value = float("inf")
            elif string_var == "-inf":
                final_value = float("-inf")
            else:
                with contextlib.suppress(ValueError):
                    final_value = float(string_var)

            if is_dict:
                obj[name] = final_value
            else:
                setattr(obj, name, final_value)

            self.__call_var_traces(name)

        return update

    def __create_vars(self, obj):
        """
        Replaces __create_vars, but uses our PySide custom Var classes 
        instead of tk.StringVar / tk.BooleanVar, etc.
        """
        new_vars = {}
        is_dict = isinstance(obj, dict)
        is_config = isinstance(obj, BaseConfig)

        if is_config:
            # obj.types is the dict of {field_name -> type}, from your original BaseConfig
            for name, var_type in obj.types.items():
                obj_var = getattr(obj, name)
                nullable = obj.nullables[name]

                if issubclass(var_type, BaseConfig):
                    # Recurse
                    var = UIState(self.parent, obj_var)
                    new_vars[name] = var
                elif var_type is str:
                    var = _StringVar(obj_var if obj_var is not None else "", nullable)
                    # attach the callback that updates the underlying object
                    cb = self.__set_str_var(obj, is_dict, name, var, nullable)
                    _id = var.trace_add(cb)
                    new_vars[name] = var
                elif issubclass(var_type, Enum):
                    var = _StringVar(str(obj_var) if obj_var else "", nullable)
                    cb = self.__set_enum_var(obj, is_dict, name, var_type=var_type, var=var, nullable=nullable)
                    _id = var.trace_add(cb)
                    new_vars[name] = var
                elif var_type is bool:
                    var = _BoolVar(bool(obj_var), nullable)
                    cb = self.__set_bool_var(obj, is_dict, name, var)
                    _id = var.trace_add(cb)
                    new_vars[name] = var
                elif var_type is int:
                    default_str = str(obj_var) if obj_var is not None else ""
                    var = _StringVar(default_str, nullable)
                    cb = self.__set_int_var(obj, is_dict, name, var, nullable)
                    _id = var.trace_add(cb)
                    new_vars[name] = var
                elif var_type is float:
                    default_str = str(obj_var) if obj_var is not None else ""
                    var = _StringVar(default_str, nullable)
                    cb = self.__set_float_var(obj, is_dict, name, var, nullable)
                    _id = var.trace_add(cb)
                    new_vars[name] = var

        else:
            # if it's just a dict or a plain object
            iterable = obj.items() if is_dict else vars(obj).items()
            for name, obj_var in iterable:
                if isinstance(obj_var, str):
                    var = _StringVar(obj_var, False)
                    cb = self.__set_str_var(obj, is_dict, name, var, False)
                    var.trace_add(cb)
                    new_vars[name] = var
                elif isinstance(obj_var, Enum):
                    var = _StringVar(str(obj_var), False)
                    cb = self.__set_enum_var(obj, is_dict, name, var, type(obj_var), False)
                    var.trace_add(cb)
                    new_vars[name] = var
                elif isinstance(obj_var, bool):
                    var = _BoolVar(obj_var, False)
                    cb = self.__set_bool_var(obj, is_dict, name, var)
                    var.trace_add(cb)
                    new_vars[name] = var
                elif isinstance(obj_var, int):
                    var = _StringVar(str(obj_var), False)
                    cb = self.__set_int_var(obj, is_dict, name, var, False)
                    var.trace_add(cb)
                    new_vars[name] = var
                elif isinstance(obj_var, float):
                    var = _StringVar(str(obj_var), False)
                    cb = self.__set_float_var(obj, is_dict, name, var, False)
                    var.trace_add(cb)
                    new_vars[name] = var
                # If none of the above, we skip it.

        return new_vars

    def __set_vars(self, obj):
        """
        Re-syncs each variable object with the new data in obj.
        This parallels your old __set_vars logic.
        """
        is_dict = isinstance(obj, dict)
        is_config = isinstance(obj, BaseConfig)

        if is_config:
            for name, var_type in obj.types.items():
                obj_var = getattr(obj, name)
                var = self.__vars[name]
                if issubclass(var_type, BaseConfig):
                    # Recurse
                    var.__set_vars(obj_var)
                elif var_type is str:
                    var.set("" if obj_var is None else obj_var)
                elif issubclass(var_type, Enum):
                    var.set("" if obj_var is None else str(obj_var))
                elif var_type is bool:
                    var.set(bool(obj_var))
                elif var_type in (int, float):
                    var.set("" if obj_var is None else str(obj_var))
        else:
            iterable = obj.items() if is_dict else vars(obj).items()
            for name, obj_var in iterable:
                if name not in self.__vars:
                    continue
                var = self.__vars[name]
                if isinstance(obj_var, str):
                    var.set(obj_var)
                elif isinstance(obj_var, Enum):
                    var.set(str(obj_var))
                elif isinstance(obj_var, bool):
                    var.set(bool(obj_var))
                elif isinstance(obj_var, (int, float)):
                    var.set(str(obj_var))
