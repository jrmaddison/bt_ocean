import jax
import json
import keras
try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None
import pathlib
import pytest

import runpy


def run_example_notebook(filename, tmp_path):
    tmp_filename = tmp_path / "tmp.py"

    with open(filename, encoding="utf-8") as nb_h, \
            open(tmp_filename, "w", encoding="utf-8") as py_h:
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


@pytest.mark.skipif(matplotlib is None, reason="matplotlib not available")
def test_0_getting_started(tmp_path):
    if not jax.config.x64_enabled:
        pytest.skip("float64 not available")

    run_example_notebook(pathlib.Path(__file__).parent.parent / "docs" / "source" / "examples" / "0_getting_started.ipynb",
                         tmp_path)


@pytest.mark.skipif(matplotlib is None, reason="matplotlib not available")
@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="Require Keras with the JAX backend")
def test_1_keras_integration(tmp_path):
    if not jax.config.x64_enabled:
        pytest.skip("float64 not available")

    run_example_notebook(pathlib.Path(__file__).parent.parent / "docs" / "source" / "examples" / "1_keras_integration.ipynb",
                         tmp_path)


@pytest.mark.skipif(matplotlib is None, reason="matplotlib not available")
def test_2_steady_state(tmp_path):
    if not jax.config.x64_enabled:
        pytest.skip("float64 not available")

    run_example_notebook(pathlib.Path(__file__).parent.parent / "docs" / "source" / "examples" / "2_steady_state.ipynb",
                         tmp_path)
