import inspect

print("=" * 100)
print("DAMAGE TOOLS MODULE")
print("=" * 100)

import hermes.tools.damage_tools as dt

print("MODULE:")
print(dt.__file__)

print("\nPUBLIC CLASSES:")
for name, obj in inspect.getmembers(dt, inspect.isclass):
    if obj.__module__ == dt.__name__:
        print(f"  {name}")

print("\nPUBLIC FUNCTIONS:")
for name, obj in inspect.getmembers(dt, inspect.isfunction):
    if obj.__module__ == dt.__name__:
        try:
            print(f"  {name}{inspect.signature(obj)}")
        except Exception:
            print(f"  {name}")

print("\nTOOL-LIKE OBJECTS:")
for name in dir(dt):
    obj = getattr(dt, name)
    if hasattr(obj, "name") or hasattr(obj, "func") or hasattr(obj, "args_schema"):
        print(f"\n{name}:")
        print(f"  type: {type(obj)}")
        print(f"  name: {getattr(obj, 'name', None)}")
        print(f"  args_schema: {getattr(obj, 'args_schema', None)}")
        if hasattr(obj, "func"):
            print(f"  func: {obj.func}")
            try:
                print(f"  func signature: {inspect.signature(obj.func)}")
            except Exception:
                pass

print("\n" + "=" * 100)
print("GEO DAMAGE ANALYSIS CORE MODULE")
print("=" * 100)

import hermes.core.geo_damage_analysis as gda

print("MODULE:")
print(gda.__file__)

print("\nPUBLIC CLASSES:")
for name, obj in inspect.getmembers(gda, inspect.isclass):
    if obj.__module__ == gda.__name__:
        print(f"  {name}")
        try:
            print(f"    init: {inspect.signature(obj.__init__)}")
        except Exception:
            pass
        for method_name in ["run", "analyze", "predict", "classify", "_run"]:
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                try:
                    print(f"    {method_name}: {inspect.signature(method)}")
                except Exception:
                    print(f"    {method_name}")

print("\nPUBLIC FUNCTIONS:")
for name, obj in inspect.getmembers(gda, inspect.isfunction):
    if obj.__module__ == gda.__name__:
        try:
            print(f"  {name}{inspect.signature(obj)}")
        except Exception:
            print(f"  {name}")

print("\nMODEL DEFAULTS / MODEL STRINGS:")
source = inspect.getsource(gda)
for line_no, line in enumerate(source.splitlines(), start=1):
    lowered = line.lower()
    if "model" in lowered or "gemini" in lowered or "gpt" in lowered or "openai" in lowered:
        print(f"{line_no}: {line}")
