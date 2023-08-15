def load_shader(name):
    from importlib import resources
    with open(resources.files("topsy.shaders") / name, "r") as f:
        return f.read()
