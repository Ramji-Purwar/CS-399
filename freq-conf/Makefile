CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -mavx512f -mavx512dq \
             -pthread -Wall -Wextra -Iinclude
LDFLAGS  := -pthread

SRC_DIR   := src
TEST_DIR  := tests
BIN_DIR   := bin

SRCS := $(SRC_DIR)/topology.cpp \
        $(SRC_DIR)/timer.cpp \
        $(SRC_DIR)/msr_reader.cpp \
        $(SRC_DIR)/workload_kernel.cpp \
        $(SRC_DIR)/burst_engine.cpp \
        $(SRC_DIR)/experiment_runner.cpp

MAIN_SRC  := $(SRC_DIR)/main.cpp
TEST_SRC  := $(TEST_DIR)/test_components.cpp

TARGET    := $(BIN_DIR)/avx512_freq_characterizer
TEST_BIN  := $(BIN_DIR)/test_components

.PHONY: all test clean help

all: $(BIN_DIR) $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR) results

$(TARGET): $(SRCS) $(MAIN_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS) $(MAIN_SRC) -o $@ $(LDFLAGS)
	@echo "Built: $@"

test: $(BIN_DIR) $(TEST_BIN)
	@echo "Running tests..."
	./$(TEST_BIN)

$(TEST_BIN): $(SRCS) $(TEST_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS) $(TEST_SRC) -o $@ $(LDFLAGS)
	@echo "Built: $@"

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "Targets:"
	@echo "  all     Build the main experiment binary"
	@echo "  test    Build and run all component tests"
	@echo "  clean   Remove build artifacts"
	@echo ""
	@echo "Usage examples:"
	@echo "  # Dry-run pipeline check"
	@echo "  ./bin/avx512_freq_characterizer --dry-run --cores 1,2,4 --cycles 50"
	@echo ""
	@echo "  # Full experiment (needs msr module + permissions)"
	@echo "  sudo modprobe msr"
	@echo "  sudo chmod +r /dev/cpu/*/msr"
	@echo "  ./bin/avx512_freq_characterizer --cores 1,2,4,8,14,28 --cycles 500"
	@echo ""
	@echo "  # High-resolution single-core sweep"
	@echo "  ./bin/avx512_freq_characterizer --cores 1 --cycles 1000 --resolution 0.002"
