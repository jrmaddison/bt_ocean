import json
import pathlib
import runpy


def run_example_notebook(filename, tmp_path):
    tmp_filename = tmp_path / "tmp.py"

    with open(filename) as nb_h, open(tmp_filename, "w") as py_h:
        nb = json.load(nb_h)
        if nb["metadata"]["language_info"]["name"] != "python":
            raise RuntimeError("Expected a Python notebook")

        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                for line in cell["source"]:
                    if not line.startswith("%"):
                        py_h.write(line)
                py_h.write("\n\n")

    runpy.run_path(str(tmp_filename))


def test_0_getting_started(tmp_path):
    run_example_notebook(pathlib.Path(__file__).parent.parent / "docs" / "source" / "examples" / "0_getting_started.ipynb",
                         tmp_path)


def test_1_keras_integration(tmp_path):
    run_example_notebook(pathlib.Path(__file__).parent.parent / "docs" / "source" / "examples" / "1_keras_integration.ipynb",
                         tmp_path)
