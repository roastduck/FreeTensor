PARSERS = MyLexer.cpp MyParser.cpp
SRCS = $(sort $(wildcard *.cpp) $(PARSERS))
OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)

.PHONY: all clean

# all: main
all: MyParser.o

-include $(DEPS)

$(PARSERS) : MyLexer.g4 MyParser.g4
	java -jar ../antlr/antlr-4.7.1-complete.jar -Werror -Dlanguage=Cpp -no-listener -no-visitor -o . MyLexer.g4 MyParser.g4

%.o: %.cpp $(PARSERS)
	g++ -std=c++17 -c -g -Wall $< -o $@ -MD -MF $*.d -MP -Wno-attributes -fPIC -I../antlr/antlr4-cpp-runtime/runtime/src -I../../include/

main: $(OBJS)
	g++ $(OBJS) ../antlr/antlr4-cpp-runtime/dist/libantlr4-runtime.a  -no-pie -o $@

clean:
	rm -f *.d *.o $(PARSERS) $(PARSERS:.cpp=.h) $(PARSERS:.cpp=.interp) $(PARSERS:.cpp=.tokens) main

