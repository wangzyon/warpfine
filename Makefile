cc        := g++
name      := pro
run_args  := yq.jpg
workdir   := workspace
srcdir    := src
objdir    := objs
stdcpp    := c++11
cuda_home := /usr/local/anaconda3/lib/python3.8/site-packages/trtpy/trt8cuda115cudnn8
syslib    := /usr/local/anaconda3/lib/python3.8/site-packages/trtpy/lib
cpp_pkg   := /usr/local/anaconda3/lib/python3.8/site-packages/trtpy/cpp-packages
cuda_arch := 
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc)


# 配置待编译的cpp文件项和依赖项mk文件
cpp_srcs := $(wildcard src/*.cpp)                            
cpp_objs := $(patsubst src/%.cpp, objs/%.cpp.o, $(cpp_srcs))         
cpp_mks := $(objs:.o=.mk)                                    


# 配置待编译的cu文件项和依赖项mk文件
cu_srcs := $(wildcard src/*.cu)                            
cu_objs := $(patsubst src/%.cu, objs/%.cu.o, $(cu_srcs))         
cu_mks := $(objs:.o=.mk) 

# 配置编译时头文件查询目录
include_paths := src              \
    $(cuda_home)/include/cuda     \
	$(cuda_home)/include/tensorRT \
	$(cuda_home)/include/protobuf \
	$(cpp_pkg)/opencv4.2/include


# 配置链接时动态库、静态库查询目录
library_paths := $(cuda_home)/lib64 \
	$(syslib)                       \
	$(cpp_pkg)/opencv4.2/lib


# 配置参与链接的动态库和静态库
link_cuda      := cudart
link_protobuf  := 
link_tensorRT  := 
link_opencv    := opencv_core opencv_imgproc opencv_imgcodecs  
link_sys       := stdc++ dl
link_librarys  := $(link_cuda) $(link_tensorRT) $(link_sys) $(link_opencv)


# 配置系统环境变量LD_LIBRARY_PATH，补充运行时动态库查询目录，针对参与链接的动态库或有其他依赖库的场景
empty := 
ld_library_path := $(subst $(empty) $(empty),:,$(library_paths))
export LD_LIBRARY_PATH:=/usr/local/anaconda3/lib/python3.8/site-packages/trtpy/cpp-packages/opencv4.2/lib

# 把库路径和头文件路径拼接起来成一个，批量自动加-Wl、-I、-L、-l
run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))



# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
# 显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
# 如果是 jetson nano，提示找不到-m64指令，请删掉 -m64选项。不影响结果
cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread
cu_compile_flags  := -std=$(stdcpp) -w -g -O0 -m64 $(cuda_arch) -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)


# 非clean指令时，将mks文件中的<目标>:<前置项>包含进来
# -include区别于include,当不存在mk文件,但存在mk目标时,将触发mk目标生成来避免错误
# 如果头文件修改了，这里的指令可以让他自动编译依赖的cpp或者cu文件
ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

# 编译cpp
$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

# 编译cu
$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)


# 编译cpp依赖项，生成mk文件
$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
    
# 编译cu文件的依赖项，生成cumk文件
$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

# 链接
$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)



$(name)   : $(workdir)/$(name)
	@echo Compile Link completed

all       : $(name)
run       : $(name)
	@cd $(workdir) && ./$(name) $(run_args)


# 定义清理指令
clean :
	@rm -rf $(objdir) $(workdir)/$(name)

# 伪目标，表示指令
.PHONY : clean run $(name) all