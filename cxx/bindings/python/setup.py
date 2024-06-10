import os
import re
import subprocess
import sys
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "Win64",
    "win-arm32": "ARM32",
    "win-arm64": "ARM64",
}

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        
class CMakeBuild(build_ext):
    # def __init__(self, ext: CMakeExtension) -> None:
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        
        debug = int(os.environ.get("DEBUG", "0") if self.debug is None else self.debug)
        cfg = "Debug" if debug else "Release"
        
        cmake_generator = os.environ.get("CmakeGenerator", "")
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}", # not used on MSVC, but no harm
        ]
        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [arg for arg in os.environ["CMAKE_ARGS"].split(" ") if arg]
        
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]
        
        if self.compiler.compiler_type != "msvc":
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja
                    
                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM={ninja_executable_path}"
                    ]
                except ImportError:
                    pass
                
        else:
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})
            
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})
            
            if not single_config:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
                
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]
                
        if sys.platform == "darwin":
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]
                
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                cmake_args += [f"-j{self.parallel}"]
                
        build_temp = Path(self.build_temp)
        if not build_temp.exists():
            build_temp.mkdir(parents=True)
        
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )
        
setup(
    name="cinnamon",
    version="0.1.0",
    author="Pythera",
    author_email="nguyenvanthanhdat1810@gmail.com",
    description="C++ Python Binding Cinamon",
    long_description="",
    ext_modules=[CMakeExtension("cinnamon")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.7"
    
)