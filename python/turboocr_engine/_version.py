"""Single source of truth for the package version.

PEP 440 spelling of the version CMakeLists.txt and the docs call
4.0.0-alpha.5. scikit-build-core reads this file by regex
(pyproject.toml [tool.scikit-build.metadata.version]), so the wheel,
`turboocr --version` and `turboocr doctor` all follow from this line.
"""

__version__ = "4.0.0a5"
