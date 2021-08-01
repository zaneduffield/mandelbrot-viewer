from .cli import make_config
from brot.ui.tkinter_ui import run


def main():
    run(make_config())


if __name__ == "__main__":
    main()
