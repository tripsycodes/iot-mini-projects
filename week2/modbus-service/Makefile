CXX := g++
CFLAGS := -O0 -Wall -I.

%.o: %.cpp
	$(CXX) $(CFLAGS) -o $@ -c $<

main: main.o ModbusMaster.o stream.o
	$(CXX) $(CFLAGS) -o $@ $^

all: main

clean:
	rm -rf *.o main

.PHONY := clean all
.DEFAULT_TARGET := all
