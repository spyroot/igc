from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    # keep only real requirement lines: comments and blanks are documentation.
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.strip().startswith('#')
    ]

setup_info = dict(
    name='igc',
    version='0.1',
    author='Mustafa Bayramov',
    author_email="spyroot@gmail.com",
    url="https://github.com/spyroot/igc",
    description='Infrastructure Goal Condition Reinforce Learner.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['igc'] + ['igc.' + pkg for pkg in find_packages('igc')],
    license="MIT",
    # <3.13 until shared_torch_utils drops distutils.LooseVersion.
    python_requires='>=3.10,<3.13',
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7",
            "ruff",
        ],
        # cluster-only: ZeRO sharding; fsdp needs no extra dependency.
        "deepspeed": [
            "deepspeed>=0.14",
        ],
    },
)
setup(**setup_info)
