import inspect

from hermes.tools.perception_tools import geo_perception_tool

tool = geo_perception_tool

print("OBJECT:")
print(tool)

print("\nTYPE:")
print(type(tool))

print("\nAVAILABLE CALL/RUN METHODS:")
for name in dir(tool):
    if "run" in name.lower() or "call" in name.lower() or "invoke" in name.lower():
        attr = getattr(tool, name)
        if callable(attr):
            try:
                print(f"  {name}{inspect.signature(attr)}")
            except Exception:
                print(f"  {name}")

print("\nNAME:")
print(getattr(tool, "name", None))

print("\nDESCRIPTION:")
print(getattr(tool, "description", None))

print("\nARGS SCHEMA:")
print(getattr(tool, "args_schema", None))

print("\nHAS FUNC:")
print(hasattr(tool, "func"))
if hasattr(tool, "func"):
    print(tool.func)
    print(inspect.signature(tool.func))
