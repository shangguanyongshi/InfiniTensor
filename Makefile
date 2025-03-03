# 在 Makefile 中，目标（target）通常对应于一个实际的生成的可文件，当执行 make target_name 命令时，
# Makefile 会根据目标的依赖关系和规则来生成目标文件，
# 同时会根据时间戳决定是否需要重新执行目标对应的操作生成目标文件。
# 但是有些非生成可执行文件的目标，如编译、清理、格式化代码等，是不需要生成文件，也不需要根据时间戳
# 判断是否要重新执行对应的操作，而是必须要执行，这些目标称为伪目标。
# 通过 .PHONY 显式指定伪目标，避免执行时间戳检查的逻辑，还可以避免存在同名的文件导致该目标对应的操作未成功执行。
.PHONY : build clean format install-python test-cpp test-onnx install-infini

TYPE ?= Release
CUDA ?= OFF
BANG ?= OFF
KUNLUN ?= OFF
ASCEND ?= OFF
INTELCPU ?= off
BACKTRACE ?= ON
TEST ?= ON
DIST ?= OFF
NNET ?= OFF
DIST ?= OFF
FORMAT_ORIGIN ?=
# Docker build options
DOCKER_NAME ?= infinitensor
DOCKER_IMAGE_NAME ?= infinitensor
DOCKER_FILE ?= infinitensor_ubuntu_22.04.dockerfile
DOCKER_RUN_OPTION ?=

# CUDA option.
ifeq ($(CUDA), ON)
	DOCKER_IMAGE_NAME = infinitensor_cuda
	DOCKER_NAME = infinitensor_cuda
	DOCKER_FILE = infinitensor_ubuntu_22.04_CUDA.dockerfile
	DOCKER_RUN_OPTION += --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`:`pwd` -w `pwd`
endif

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += -DUSE_CUDA=$(CUDA)
CMAKE_OPT += -DUSE_BANG=$(BANG)
CMAKE_OPT += -DUSE_KUNLUN=$(KUNLUN)
CMAKE_OPT += -DUSE_ASCEND=$(ASCEND)
CMAKE_OPT += -DUSE_BACKTRACE=$(BACKTRACE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)
CMAKE_OPT += -DBUILD_DIST=$(DIST)
CMAKE_OPT += -DBUILD_NNET=$(NNET)

ifeq ($(INTELCPU), ON)
	CMAKE_OPT += -DUSE_INTELCPU=ON -DCMAKE_CXX_COMPILER=dpcpp
endif

build:
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j8

clean:
	rm -rf build

# @ 符号的作用是在执行命令时不显示命令本身，只显示命令的输出。
# 如果没有 @ 符号，Makefile 会在执行命令之前先打印出命令内容
format:
	@python3 scripts/format.py $(FORMAT_ORIGIN)

install-python: build
	cp build/$(TYPE)/backend*.so pyinfinitensor/src/pyinfinitensor
	pip install -e pyinfinitensor/

# 克隆、编译并安装 infinop 统一算子库
install-infini:
	git clone -b dev https://github.com/PanZezhong1725/operators.git --recursive; \
	cd operators && \
	if [ "$(CUDA)" = "ON" ]; then \
		xmake f --nv-gpu=true --cuda=$(CUDA_HOME) -cv; \
	fi && \
	xmake build && xmake install

# CTest 的 add_test 会生成 test 目标，因此使用 make test 和 ctest 命令执行测试用例是等价的
test-cpp:
	@echo
	cd build/$(TYPE) && make test

# 执行 pyinfinitensor/tests/test_onnx.py 文件
test-onnx:
	@echo
	python3 pyinfinitensor/tests/test_onnx.py

# 执行 pyinfinitensor/tests/test_api.py 文件
test-api:
	@echo
	python3 pyinfinitensor/tests/test_api.py

# docker 相关的命令
docker-build:
	docker build -f scripts/dockerfile/$(DOCKER_FILE) -t $(DOCKER_NAME) .

docker-run:
	docker run -t --name $(DOCKER_IMAGE_NAME) -d $(DOCKER_NAME) $(DOCKER_RUN_OPTION)

docker-start:
	docker start $(DOCKER_IMAGE_NAME)

docker-exec:
	docker exec -it $(DOCKER_IMAGE_NAME) bash
