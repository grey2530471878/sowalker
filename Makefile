
CC = g++
INCLUDE = -I.
FLAGS = -std=c++11  -lortools -g -fopenmp -w -lpthread -D FASTSKIP -D IO_UTE   

apps : test/node2vec

test/% : test/%.cpp
	@mkdir -p bin/$(@D)
	$(CC) $@.cpp -o bin/$@ $(INCLUDE) $(FLAGS)

clean :
	-rm -rf bin

clear : 
	-rm dataset/*.deg dataset/*.csr dataset/*.beg dataset/*.rat dataset/*.blocks dataset/*.meta

walks:
	-rm dataset/*.walk
