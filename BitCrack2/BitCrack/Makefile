#---------------------------------------------------------------------
# Makefile for BitCrack
#

SRC =   AddrUtil/Base58.cpp                \
        AddrUtil/hash.cpp                  \
        CmdParse/CmdParse.cpp              \
        CryptoUtil/checksum.cpp            \
        CryptoUtil/ripemd160.cpp           \
        CryptoUtil/Rng.cpp                 \
        CryptoUtil/sha256.cpp              \
        CudaDevice/CudaKeySearchDevice.cpp \
        CudaUtil/cudaUtil.cpp              \
        KeyFinder/ConfigFile.cpp           \
        KeyFinder/DeviceManager.cpp        \
        KeyFinder/KeyFinder.cpp            \
        Logger/Logger.cpp                  \
        main.cpp                           \
        Secp256k1/secp256k1.cpp            \
        Util/util.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/,            \
        AddrUtil/Base58.o                  \
        AddrUtil/hash.o                    \
        CmdParse/CmdParse.o                \
        CryptoUtil/checksum.o              \
        CryptoUtil/ripemd160.o             \
        CryptoUtil/Rng.o                   \
        CryptoUtil/sha256.o                \
        CudaDevice/CudaKeySearchDevice.o   \
        CudaUtil/cudaUtil.o                \
        KeyFinder/ConfigFile.o             \
        KeyFinder/DeviceManager.o          \
        KeyFinder/KeyFinder.o              \
        Logger/Logger.o                    \
        main.o                             \
        Secp256k1/secp256k1.o              \
        Util/util.o                        \
        CudaDevice/CudaAtomicList.cu.o     \
        CudaDevice/cudabridge.cu.o         \
        CudaDevice/CudaDeviceKeys.cu.o     \
        CudaDevice/CudaHashLookup.cu.o     \
        CudaDevice/CudaKeySearchDevice.cu.o\
        CudaDevice/BitCrack-link.o)

CXX        = g++
CUDA       = /usr/local/cuda-11.0
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc
# nvcc requires joint notation w/o dot, i.e. "5.2" -> "52"
ccap       = $(shell echo $(CCAP) | tr -d '.')

CXXFLAGS   = -m64 -mssse3 -Wno-write-strings -O2 -I. -I$(CUDA)/include
LFLAGS     = -lgmp -lpthread -L$(CUDA)/lib64 -lcudart -lcudadevrt


#--------------------------------------------------------------------

$(OBJDIR)/CudaDevice/CudaAtomicList.cu.o: CudaDevice/CudaAtomicList.cu
	$(NVCC) -rdc=true -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/CudaDevice/CudaAtomicList.cu.o -c CudaDevice/CudaAtomicList.cu

$(OBJDIR)/CudaDevice/cudabridge.cu.o: CudaDevice/cudabridge.cu
	$(NVCC) -rdc=true -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/CudaDevice/cudabridge.cu.o -c CudaDevice/cudabridge.cu

$(OBJDIR)/CudaDevice/CudaDeviceKeys.cu.o: CudaDevice/CudaDeviceKeys.cu
	$(NVCC) -rdc=true -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/CudaDevice/CudaDeviceKeys.cu.o -c CudaDevice/CudaDeviceKeys.cu

$(OBJDIR)/CudaDevice/CudaHashLookup.cu.o: CudaDevice/CudaHashLookup.cu
	$(NVCC) -rdc=true -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/CudaDevice/CudaHashLookup.cu.o -c CudaDevice/CudaHashLookup.cu

$(OBJDIR)/CudaDevice/CudaKeySearchDevice.cu.o: CudaDevice/CudaKeySearchDevice.cu
	$(NVCC) -rdc=true -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/CudaDevice/CudaKeySearchDevice.cu.o -c CudaDevice/CudaKeySearchDevice.cu

$(OBJDIR)/CudaDevice/BitCrack-link.o: $(OBJDIR)/CudaDevice/CudaAtomicList.cu.o $(OBJDIR)/CudaDevice/cudabridge.cu.o $(OBJDIR)/CudaDevice/CudaDeviceKeys.cu.o $(OBJDIR)/CudaDevice/CudaHashLookup.cu.o $(OBJDIR)/CudaDevice/CudaKeySearchDevice.cu.o
	$(NVCC) -link -maxrregcount=0 --ptxas-options=-v --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/CudaDevice/BitCrack-link.o -dlink $(OBJDIR)/CudaDevice/CudaAtomicList.cu.o $(OBJDIR)/CudaDevice/cudabridge.cu.o $(OBJDIR)/CudaDevice/CudaDeviceKeys.cu.o $(OBJDIR)/CudaDevice/CudaHashLookup.cu.o $(OBJDIR)/CudaDevice/CudaKeySearchDevice.cu.o -L$(CUDA)/lib64 -lcudart -lcudadevrt


$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: BitCrack

BitCrack: $(OBJET)
	@echo Making BitCrack...
	
	$(CXX) $(OBJET) $(LFLAGS) -o BitCrack

$(OBJET): | $(OBJDIR) $(OBJDIR)/AddrUtil $(OBJDIR)/CmdParse $(OBJDIR)/CryptoUtil $(OBJDIR)/CudaUtil $(OBJDIR)/KeyFinder $(OBJDIR)/Logger $(OBJDIR)/Secp256k1 $(OBJDIR)/Util $(OBJDIR)/CudaDevice

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/AddrUtil: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p AddrUtil

$(OBJDIR)/CmdParse: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p CmdParse

$(OBJDIR)/CryptoUtil: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p CryptoUtil

$(OBJDIR)/CudaUtil: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p CudaUtil

$(OBJDIR)/KeyFinder: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p KeyFinder

$(OBJDIR)/Logger: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p Logger

$(OBJDIR)/Secp256k1: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p Secp256k1

$(OBJDIR)/Util: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p Util

$(OBJDIR)/CudaDevice: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p CudaDevice

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/AddrUtil/*.o
	@rm -f obj/CmdParse/*.o
	@rm -f obj/CryptoUtil/*.o
	@rm -f obj/CudaUtil/*.o
	@rm -f obj/KeyFinder/*.o
	@rm -f obj/Logger/*.o
	@rm -f obj/Secp256k1/*.o
	@rm -f obj/Util/*.o
	@rm -f obj/CudaDevice/*.o

