all: cop

cop: cop.cpp
	g++ -O3 -std=c++11 -o cop cop.cpp

