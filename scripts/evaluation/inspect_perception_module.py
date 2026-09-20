import inspect
import hermes.tools.perception_tools as pt

print("MODULE:")
print(pt.__file__)

print("\nPUBLIC CLASSES:")
for name, obj in inspect.getmembers(pt, inspect.isclass):
    if obj.__module__ == pt.__name__:
        print(f"  {name}")

print("\nPUBLIC FUNCTIONS:")
for name, obj in inspect.getmembers(pt, inspect.isfunction):
    if obj.__module__ == pt.__name__:
        print(f"  {name}{inspect.signature(obj)}")
