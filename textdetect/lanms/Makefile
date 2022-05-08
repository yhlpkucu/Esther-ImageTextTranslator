#CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3-config --cflags)
CXXFLAGS = -I include  -std=c++11 -O3 -I/home/yhl/anaconda3/include/python3.7m -I/home/yhl/anaconda3/include/python3.7m  -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -O3 -pipe  -fdebug-prefix-map==/usr/local/src/conda/- -fdebug-prefix-map==/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -flto -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -DNDEBUG -fwrapv -O3 -Wall

#LDFLAGS = $(shell python3-config --ldflags)
LDFLAGS = -L/home/yhl/anaconda3/lib/python3.7/config-3.7m-x86_64-linux-gnu -L/home/yhl/anaconda3/lib -lpython3.7m -lpthread -ldl  -lutil -lrt -lm  -Xlinker -export-dynamic


DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
