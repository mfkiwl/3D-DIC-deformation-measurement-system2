# 定義編譯器
CC = gcc

# 定義編譯參數 (-Wall 顯示所有警告, -g 加入除錯資訊, -I 從root開始找)
CFLAGS = -Wall -g -I. -fPIC

# 定義連結參數
LDFLAGS = -lm -shared -Wl,--export-all-symbols

# 定義目標執行檔名稱
TARGET = 2D_DIC.dll

# 定義所有的來源檔案 (自動找.c)
SRCS = $(wildcard *.c) \
       $(wildcard common/*.c) \
       $(wildcard core/*.c) \
       $(wildcard factory/*.c) \
       $(wildcard image/*.c)

# 預設規則
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# 清除編譯結果
clean:
	rm -f $(TARGET)