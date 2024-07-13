class Settings:
    ...

def language(lang: str):
    ...

def gnu_extensions(enabled: bool):
    ...

def target(value: str):
    if value == "host":
        # use host architecture
        ...

def standard(selector: str):
    ...

def standards(selector: str):
    ...

# TODO compiler specific settings