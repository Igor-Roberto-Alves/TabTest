"""Cria um modelo novo a partir do molde: python -m src.models.new <nome>"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main():
    if len(sys.argv) != 2 or not re.fullmatch(r"[a-z][a-z0-9_]*", sys.argv[1]):
        sys.exit(__doc__)
    name = sys.argv[1]
    dest = HERE / f"{name}.py"
    if dest.exists():
        sys.exit(f"{dest} já existe")
    dest.write_text((HERE / "_template.py").read_text().replace("__NAME__", name))
    print(f"criado {dest}; implemente fit_predict e rode: python -m src.run --task iris --models {name}")


if __name__ == "__main__":
    main()
