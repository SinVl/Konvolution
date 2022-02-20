from setuptools import setup, find_packages
from os.path import join, dirname

try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements


def load_requirements(file_name):
    reqs = parse_requirements(file_name, session="test")
    try:
        requirements = [str(ir.req) for ir in reqs]
    except:
        requirements = [str(ir.requirement) for ir in reqs]
    return requirements


setup(
    name='Konvolution',
    version='1.0.0',
    author="Sineglazov Vladislav",
    author_email="<sineglazov.v@yandex.ru>",
    packages=['konvolution', ''],
    install_requires=load_requirements("requirements.txt"),
    include_package_data=True,
)
