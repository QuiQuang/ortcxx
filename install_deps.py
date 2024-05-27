import os
import sys
import tarfile
import requests
import argparse
import platform
from tqdm import tqdm
from pathlib import Path


def download(desc: str, url: str, out: str):
  """
  Download file.
  :param desc:  description info.
  :param url:   URL to be downloaded.
  :param out:   output file path.
  """
  if not os.path.isfile(out):
    res = requests.get(url=url, stream=True)
    bsize, tsize = 1024, int(res.headers.get('content-length', 0))
    with tqdm(desc=desc, total=tsize, unit='B', unit_scale=True) as pbar:
      with open(out, 'wb') as file:
        for data in res.iter_content(bsize):
          pbar.update(len(data))
          file.write(data)
    if tsize != 0 and pbar.n != tsize:
      raise RuntimeError('Could not download file.')


def tarextract(src: str, dst: str):
  """
  Extract TAR file.
  :param src:   source file to be extracted.
  :param dst:   extract destination.
  """
  with tarfile.open(src) as tar:
    tar.extractall(dst)


def cuda_float8(cu) -> bool:
  """
  Check if cuda support new type float8.
  :param cu:  cuda version string.
  """
  _cu = [int(x) for x in cu.split('.')]
  if _cu[0] < 12:
    return False
  elif _cu[1] < 8:
    return False
  return True


def ort(_os: str, deps: Path, args):
  """
  Build and install OnnxRuntime from source.
  :param dir:     deps directory.
  :param _os:     operation system name.
  :param args:    additional arguments.
  """
  print('Install OnnxRuntime from source...')
  release = 'Release'
  url = f'https://github.com/microsoft/onnxruntime/archive/refs/tags/v{args.v}.tar.gz'
  tar = deps.joinpath(f'onnxruntime-{args.v}.tar.gz')
  dir = deps.joinpath(f'onnxruntime-{args.v}').resolve()
  build = deps.joinpath(f'onnxruntime-{args.v}', 'build', _os, release).resolve()
  # Download and extract source.
  download(f'Download {url}', url, tar)
  tarextract(tar, deps)
  # Build command.
  script = './build.bat' if _os == 'Windows' else './build.sh'
  cmd = [
    f'cd {dir} &&',
    f'bash {script} --config {release} --build_shared_lib --build_wheel',
    f'--parallel {int(os.cpu_count())} --nvcc_threads 1',
    f'--compile_no_warning_as_error --skip_submodule_sync --allow_running_as_root'
  ]
  # Skip tests.
  if not args.tests:
    cmd.append('--skip_tests')
  # Enable CUDA.
  if args.gpu:
    # Use conda CUDA
    os.system(' '.join([
      'mamba env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib',
      '&& mamba install -y',
        'cudnn',
        f'cuda-nvcc~={args.cuda}',
        f'cuda-cudart~={args.cuda}',
        f'cuda-runtime~={args.cuda}',
        f'cuda-libraries~={args.cuda}',
        f'cuda-libraries-dev~={args.cuda}',
        '-c nvidia'
    ]))
    cmd.append(f'--use_cuda --cuda_version {args.cuda}')
    cmd.append('--cuda_home ${CONDA_PREFIX} --cudnn_home ${CONDA_PREFIX}')
    # Check type compatible.
    if not cuda_float8(args.cuda):
      cmd.append('--disable_types float8')
  # Cmake extras.
  cmd.append(' '.join([
    '--cmake_extra_defines',
      'CMAKE_OSX_ARCHITECTURES=$(arch)',
      'CMAKE_CUDA_ARCHITECTURES="75;80;86"',
      'CMAKE_INSTALL_PREFIX=${CONDA_PREFIX}',
      'CMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib/cmake',
    f'&& cd {build} && make -j {os.cpu_count()} && pip install dist/* && make install'
  ]))
  # Run build command.
  os.system(' '.join(cmd))


def cv(_os, deps: Path, args):
  """
  Build and install OpenCV from source.
  :param dir:     deps directory.
  :param _os:     operation system name.
  :param args:    additional arguments.
  """
  # Download and extract source.
  url = f'https://github.com/opencv/opencv/archive/refs/tags/{args.v}.tar.gz'
  tar = deps.joinpath(f'opencv-{args.v}.tar.gz')
  dir = deps.joinpath(f'opencv-{args.v}')
  build = dir.joinpath('build')
  download(f'Download {url}', url, tar)
  tarextract(tar, deps)
  build.mkdir(parents=True, exist_ok=True)
  # Download and extract contrib source.
  contrib_url = f'https://github.com/opencv/opencv_contrib/archive/refs/tags/{args.v}.tar.gz'
  contrib_tar = deps.joinpath(f'opencv-contrib-{args.v}.tar.gz')
  contrib_dir = deps.joinpath(f'opencv_contrib-{args.v}', 'modules')
  download(f'Download {contrib_url}', contrib_url, contrib_tar)
  tarextract(contrib_tar, deps)
  # Build command.
  cmd = [
    f'cd {build} &&',
    'cmake ..',
      f'-DOPENCV_EXTRA_MODULES_PATH={contrib_dir}',
      '-DBUILD_TESTS=OFF',
      '-DBUILD_PERF_TESTS=OFF',
      '-DBUILD_EXAMPLES=OFF',
      '-DBUILD_opencv_apps=OFF'
      '-DPYTHON_DEFAULT_EXECUTABLE=$(which python)',
      '-DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}',
      '-DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib/cmake'
    f'&& make -j {os.cpu_count()} && make install'
  ]
  os.system(' '.join(cmd))


parser = argparse.ArgumentParser('Cinnamon Runtime install dependencies')
subparsers = parser.add_subparsers(help='deps')
# OnnxRuntime.
ortparser = subparsers.add_parser('ort')
ortparser.add_argument('--v', default='1.18.0', help='Onnxruntime version.')
ortparser.add_argument('--gpu', default=False, help='Enable GPU.', action=argparse.BooleanOptionalAction)
ortparser.add_argument('--cuda', default='11.8', help='CUDA version.')
ortparser.add_argument('--cudnn', default='8', help='CuDNN version.')
ortparser.add_argument('--tests', default=False, help='Build tests.', action=argparse.BooleanOptionalAction)
ortparser.set_defaults(func=ort)
# OpenCV.
cvparser = subparsers.add_parser('cv')
cvparser.add_argument('--v', default='4.9.0', help='OpenCV version.')
cvparser.set_defaults(func=cv)


if __name__ == '__main__':
  args = parser.parse_args()
  # Get OS name.
  _os = platform.system()
  _os = 'MacOS' if _os == 'Darwin' else _os
  # Make deps directory.
  _deps = Path(os.getcwd(), '_deps')
  _deps.mkdir(parents=True, exist_ok=True)
  # Run.
  args.func(_os, _deps, args)
