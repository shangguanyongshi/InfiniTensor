import sys
from pathlib import Path
from subprocess import run

c_style_file = [".h", ".hh", ".hpp", ".c", ".cc", ".cpp", ".cxx", ".cu", ".mlu"]
py_file = ".py"
proj_path = Path(sys.path[0]).parent


# Formats one file under project path.
def format_file(file):
    """对传入的文件进行格式化

    Args:
        file: 要格式化文件的绝对路径
    """
    file = Path(proj_path.joinpath(file))
    if file.suffix in c_style_file:
        # 如果是 c 文件，使用 clang-format 进行格式化
        run(f"clang-format -style=file -i {file}", cwd=proj_path, shell=True)
        run(f"git add {file}", cwd=proj_path, shell=True)
    elif file.suffix == py_file:
        # 如果是 python 文件，使用 black 进行格式化
        run(f"black {file}", cwd=proj_path, shell=True)
        run(f"git add {file}", cwd=proj_path, shell=True)


if len(sys.argv) == 1:
    # 如果没有传入参数，执行 git status 命令，列出所有发生改动的文件，再调用 format_file 函数进行格式化
    # Last commit.
    print("Formats git added files.")
    for line in (
        run("git status", cwd=proj_path, capture_output=True, shell=True)
        .stdout.decode()
        .splitlines()
    ):
        line = line.strip()
        # Only formats git added files.
        for pre in ["new file:", "modified:"]:
            if line.startswith(pre):
                format_file(line[len(pre) :].strip())
            break
else:
    # 如果传入了参数，指定了一个要格式化的文件，如果该文件发生了变换，对该文件执行格式化
    # Origin commit.
    origin = sys.argv[1]
    print(f'Formats changed files from "{origin}".')
    for line in (
        run(f"git diff {origin}", cwd=proj_path, capture_output=True, shell=True)
        .stdout.decode()
        .splitlines()
    ):
        diff = "diff --git "
        if line.startswith(diff):
            files = line[len(diff) :].split(" ")
            assert len(files) == 2
            assert files[0][:2] == "a/"
            assert files[1][:2] == "b/"
            format_file(files[1][2:])
