CC = gcc

# 定義編譯參數 (-Wall 顯示所有警告, -g 加入除錯資訊, -I 從root開始找)
CFLAGS = -Wall -g -I. -fPIC
LDFLAGS = -lm -shared -Wl,--export-all-symbols

OUTPUT_NAME = 2D_DIC.dll
TARGET = build/$(OUTPUT_NAME)

# 定義所有的來源檔案 (自動找.c)
SRCS = $(wildcard *.c) \
       $(wildcard new_function/common/*.c) \
       $(wildcard new_function/core/*.c) \
       $(wildcard new_function/factory/*.c) \
       $(wildcard new_function/image/*.c)

HEADERS = $(wildcard *.h) \
          $(wildcard include/*.h) \
          $(wildcard new_function/common/*.h) \
          $(wildcard new_function/core/*.h)

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)


.PHONY: all clean