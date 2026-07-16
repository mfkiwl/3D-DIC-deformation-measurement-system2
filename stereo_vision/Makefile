CC = gcc

# define params (-Wall: show warnings, -g: add debug info, -I : srearch from root)                       
# compile params
CFLAGS = -Wall -O2 -fPIC \
	-I. \
	-IPSO/include \
	-IDIC/include \
	-Itools/interpolation/include

LDFLAGS = -lm -shared -Wl,--export-all-symbols      # link params

BUILD_DIR            =      build

OUTPUT_PSO           =      PSO.dll
OUTPUT_INTERP        =      cubic_interp.dll
OUTPUT_ICGN          =      ICGN.dll

TARGET_PSO           =      build/$(OUTPUT_PSO)
TARGET_INTERP        =      build/$(OUTPUT_INTERP)
TARGET_ICGN          =      build/$(OUTPUT_ICGN)

# define all source (find .c)
SRCS_PSO             =      $(wildcard PSO/src/core/PSO_core.c) \
                            $(wildcard PSO/src/core/PSO.c) \
                            $(wildcard PSO/src/factory/PSO_factory.c) \
                            $(wildcard PSO/src/interp/interpolation_core.c) \
                            $(wildcard PSO/src/system/sys_maneger.c) \
                            $(wildcard PSO/src/system/system_comm.c) \
                            $(wildcard DIC/src/DIC_comm.c) \
                            $(wildcard DIC/src/DIC_core.c)

HEADERS_PSO          =      $(wildcard PSO/include/*.h) \
                            $(wildcard DIC/include/*.h) \
                            $(wildcard tools/interpolation/include/*.h)

SRCS_INTERP          =      $(wildcard tools/interpolation/src/cubic_interp.c)
HEADERS_INTERP       =      $(wildcard tools/interpolation/include/*.h)
SRCS_ICGN            =      $(wildcard DIC/src/ICGN_subset_warping.c)
HEADERS_ICGN         =      $(wildcard DIC/include/*.h)

check:
	@echo $(SRCS_PSO)

prepare:
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)

all: prepare $(TARGET_PSO) $(TARGET_INTERP) $(TARGET_ICGN)

$(TARGET_PSO): $(SRCS_PSO)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(TARGET_INTERP): $(SRCS_INTERP)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(TARGET_ICGN): $(SRCS_ICGN)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	@if exist $(BUILD_DIR) del /f /q $(BUILD_DIR)\*.dll

.PHONY: all clean

# $^: source files
# $@: target files