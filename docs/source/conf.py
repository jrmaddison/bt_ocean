project = "bt_ocean"

extensions = ["autoapi.extension",
              "nbsphinx",
              "numpydoc",
              "sphinx.ext.intersphinx",
              "sphinx_rtd_theme"]

autoapi_type = "python"
autoapi_dirs = ["../../bt_ocean"]
autoapi_ignore = []
autoapi_add_toctree_entry = False
autoapi_options = []

numpydoc_validation_checks = {"all", "GL08"}

html_theme = "sphinx_rtd_theme"
html_theme_options = {"display_version": False}

exclude_patterns = []
html_static_path = ["static"]
templates_path = []

html_css_files = ["custom.css"]

intersphinx_mapping = {"jax": ("https://jax.readthedocs.io/en/latest", None),
                       "zarr": ("https://zarr.readthedocs.io/en/stable", None)}
