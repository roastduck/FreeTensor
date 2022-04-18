PARSERS = MDLexer.cpp MDParser.cpp
SRCS = $(sort $(wildcard *.cpp) $(PARSERS))
OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)

.PHONY: all clean

all: main

-include $(DEPS)

$(PARSERS) : MDLexer.g4 MDParser.g4
	java -jar ../antlr/antlr-4.7.1-complete.jar -Werror -Dlanguage=Cpp -no-listener -no-visitor -o . MDLexer.g4 MDParser.g4

%.o: %.cpp $(PARSERS)
	g++ -c -g -Wall $< -o $@ -MD -MF $*.d -MP -Wno-attributes -fPIC -I../antlr/antlr4-cpp-runtime/runtime/src

main: $(OBJS)
	g++ $(OBJS) ../antlr/antlr4-cpp-runtime/dist/libantlr4-runtime.a  -no-pie -o $@

clean:
	rm -f *.d *.o $(PARSERS) $(PARSERS:.cpp=.h) $(PARSERS:.cpp=.interp) $(PARSERS:.cpp=.tokens) main

