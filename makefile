CC = gcc
CFLAGS = -Iinc -lm -std=c++11 -O3 -march=native -Wall
SOURCE=src
apps = test test_cuda

all: $(apps)

$(apps) : % : $(SOURCE)/%.cpp
	$(CC) $< -o $@ $(CFLAGS)

clean:
	rm -f $(apps)

doc:
	rm -R -f doxy/html
	doxygen doxy/config

#doxygen for windows
doc_w:
	rm -R -f doxy/html
	doxy/doxygen.exe doxy/config
	