# Diretórios de inclusão e linkagem do Boost
INCLUDES = -I/opt/homebrew/opt/boost/include
LDFLAGS = -L/opt/homebrew/opt/boost/lib
LDLIBS = -lboost_program_options

# Nome do binário e instalação
NAME = multidupehack
INSTALL_DIR = /usr/local/bin

# Compilador e flags
CXX = clang++
CXXFLAGS = -Ofast -flto -std=c++17 -DBOOST_MATH_OVERFLOW_ERROR_POLICY=ignore_error \
  -Wall -Wextra -Weffc++ -pedantic -Wno-overloaded-virtual -Wno-unused-parameter $(INCLUDES)

# Help2man
HELP2MAN = help2man -n 'List every (closed) noise-tolerant n-set satisfying given constraints and/or optimizing given measures' -N

# Código-fonte
SRC = src/utilities src/measures src/core
DEPS = $(wildcard $(SRC:=/*.h)) Parameters.h Makefile
CODE = $(wildcard $(SRC:=/*.cpp))
OBJ = $(CODE:.cpp=.o)
ALL = $(DEPS) $(CODE) $(NAME).1.gz COPYING example INSTALL README

# Regras padrão
.PHONY: man install clean dist-gzip dist-bzip2 dist-xz dist
.SILENT: $(NAME) man install clean dist-gzip dist-bzip2 dist-xz dist

# Regra de compilação de objetos
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Linkagem final
$(NAME): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)
	echo "$(NAME) built!"

# Gerar página de manual
man:
	$(HELP2MAN) ./$(NAME) | gzip > $(NAME).1.gz

# Instalação do binário e man page
install: $(NAME) $(NAME).1.gz
	mv $(NAME) $(INSTALL_DIR)
	cp $(NAME).1.gz /usr/share/man/man1/
	echo "$(NAME) installed!"

# Limpeza
clean:
	rm -f $(patsubst %,%/*.o,$(SRC)) $(patsubst %,%/*~,$(SRC)) *~ $(NAME)

# Empacotamentos
dist-gzip:
	tar --format=posix --transform 's,^,$(NAME)/,' -czf $(NAME).tar.gz $(ALL)

dist-bzip2:
	tar --format=posix --transform 's,^,$(NAME)/,' -cjf $(NAME).tar.bz2 $(ALL)

dist-xz:
	tar --format=posix --transform 's,^,$(NAME)/,' -cJf $(NAME).tar.xz $(ALL)

dist: dist-bzip2
