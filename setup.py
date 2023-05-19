from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_info = dict(name='igc',
                  version='0.1',
                  author='Mustafa Bayramov',
                  author_email="spyroot@gmail.com",
                  url="https://github.com/spyroot/igc",
                  description='Infrastructure Goal Condition  Reinforce Learner.',
                  long_description=long_description,
                  long_description_content_type='text/markdown',
                  packages=['igc'] + ['igc.' + pkg for pkg in find_packages('igc')],
                  license="MIT",
                  python_requires='>=3.10',
                  install_requires=requirements,
                  # entry_points={
                  #     'console_scripts': [
                  #         'idrac_ctl = idrac_ctl.idrac_main:idrac_main_ctl',
                  #     ]
                  # },
                  extras_require={
                      "dev": [
                          "pytest >= 3.7"
                      ]
                  },
                  )
setup(**setup_info)
