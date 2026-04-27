CC = gcc

# define params (-Wall: show warnings, -g: add debug info, -I : srearch from root)
CFLAGS = -Wall -g -I. -fPIC
LDFLAGS = -lm -shared -Wl,--export-all-symbols

BUILD_DIR            =      build
OUTPUT_PSO           =      PSO.dll
OUTPUT_INTERP        =      cubic_interp.dll
OUTPUT_ICGN          =      ICGN.dll
TARGET_PSO           =      build/$(OUTPUT_PSO)
TARGET_INTERP        =      build/$(OUTPUT_INTERP)
TARGET_ICGN          =      build/$(OUTPUT_ICGN)

# define all source (find .c)
SRCS_PSO             =      $(wildcard PSO/src/PSO_core.c) \
                            $(wildcard PSO/src/PSO.c) \
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
SRCS_ICGN            =      $(wildcard DIC/src/ICGN.c)
HEADERS_ICGN         =      $(wildcard DIC/include/*.h)

$(shell mkdir -p $(BUILD_DIR))

all: $(TARGET_PSO) $(TARGET_INTERP) $(TARGET_ICGN)

$(TARGET_PSO): $(SRCS_PSO) $(HEADERS_PSO)
	$(CC) $(CFLAGS) $(SRCS_PSO) -o $(TARGET_PSO) $(LDFLAGS)

$(TARGET_INTERP): $(SRCS_INTERP) $(HEADERS_INTERP)
	$(CC) $(CFLAGS) $(SRCS_INTERP) -o $(TARGET_INTERP) $(LDFLAGS)

$(TARGET_ICGN): $(SRCS_ICGN) $(HEADERS_ICGN)
	$(CC) $(CFLAGS) $(SRCS_ICGN) -o $(TARGET_ICGN) $(LDFLAGS)

clean:
	rm -f $(TARGET_PSO) $(TARGET_INTERP) $(TARGET_ICGN)

.PHONY: all clean