from setuptools import setup, find_namespace_packages

setup(
    name="tm_composite_nkom",
    version="0.1",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "numpy",
        "opencv-python-headless",
        "tqdm",
        "scikit-learn",
        "scikit-image",
        "matplotlib",
        "pyTsetlinMachine",
        "tensorflow",
        "loguru",
        "tmu",
        "pydantic",
        "pydantic_settings",
        "python-dotenv",
        "pycuda",
    ],
    python_requires=">=3.8",
) 