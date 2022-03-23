#include "CudaKeySearchDevice.h"
#include "../Logger/Logger.h"
#include "../Util/util.h"
#include "cudabridge.h"
#include "../AddrUtil/AddressUtil.h"

void CudaKeySearchDevice::cudaCall(cudaError_t err)
{
	if (err) {
		std::string errStr = cudaGetErrorString(err);

		throw KeySearchException(errStr);
	}
}

CudaKeySearchDevice::CudaKeySearchDevice(int device, int threads, int pointsPerThread, int blocks)
{
	cuda::CudaDeviceInfo info;
	try {
		info = cuda::getDeviceInfo(device);
		_deviceName = info.name;
	}
	catch (cuda::CudaException ex) {
		throw KeySearchException(ex.msg);
	}

	if (threads <= 0 || threads % 32 != 0) {
		throw KeySearchException("The number of threads must be a multiple of 32");
	}

	if (pointsPerThread <= 0) {
		throw KeySearchException("At least 1 point per thread required");
	}

	// Specifying blocks on the commandline is depcreated but still supported. If there is no value for
	// blocks, devide the threads evenly among the multi-processors
	if (blocks == 0) {
		if (threads % info.mpCount != 0) {
			throw KeySearchException("The number of threads must be a multiple of " + util::format("%d", info.mpCount));
		}

		_threads = threads / info.mpCount;

		_blocks = info.mpCount;

		while (_threads > 512) {
			_threads /= 2;
			_blocks *= 2;
		}
	}
	else {
		_threads = threads;
		_blocks = blocks;
	}

	_iterations = 0;

	_device = device;

	_pointsPerThread = pointsPerThread;
}

void CudaKeySearchDevice::init(const secp256k1::uint256& start, int compression, int searchMode, const secp256k1::uint256& stride)
{
	if (start.cmp(secp256k1::N) >= 0) {
		throw KeySearchException("Starting key is out of range");
	}

	_startExponent = start;

	_compression = compression;

	_searchMode = searchMode;

	_stride = stride;

	cudaCall(cudaSetDevice(_device));

	// Block on kernel calls
	cudaCall(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

	// Use a larger portion of shared memory for L1 cache
	cudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	generateStartingPoints();

	cudaCall(allocateChainBuf(_threads * _blocks * _pointsPerThread));

	// Set the incrementor
	secp256k1::ecpoint g = secp256k1::G();
	secp256k1::ecpoint p = secp256k1::multiplyPoint(secp256k1::uint256((uint64_t)_threads * _blocks * _pointsPerThread) * _stride, g);

	cudaCall(_resultList.init(sizeof(CudaDeviceResult), 16));

	cudaCall(setIncrementorPoint(p.x, p.y));
}

CudaKeySearchDevice::~CudaKeySearchDevice()
{
	_deviceKeys.clearPrivateKeys();
}

void CudaKeySearchDevice::generateStartingPoints()
{
	uint64_t totalPoints = (uint64_t)_pointsPerThread * _threads * _blocks;
	uint64_t totalMemory = totalPoints * 40;

	std::vector<secp256k1::uint256> exponents;

	Logger::log(LogLevel::Info, "Generating " + util::formatThousands(totalPoints) + " starting points (" + util::format("%.1f", (double)totalMemory / (double)(1024 * 1024)) + "MB)");

	// Generate key pairs for k, k+1, k+2 ... k + <total points in parallel - 1>
	secp256k1::uint256 privKey = _startExponent;

	exponents.push_back(privKey);

	for (uint64_t i = 1; i < totalPoints; i++) {
		privKey = privKey.add(_stride);
		exponents.push_back(privKey);
	}

	cudaCall(_deviceKeys.init(_blocks, _threads, _pointsPerThread, exponents));

	// Show progress in 10% increments
	double pct = 10.0;
	for (int i = 1; i <= 256; i++) {
		cudaCall(_deviceKeys.doStep());

		if (((double)i / 256.0) * 100.0 >= pct) {
			if (pct <= 10.0)
				Logger::log2(LogLevel::Info, util::format("%.1f%%", pct));
			else
				printf("  %.1f%%", pct);
			pct += 10.0;
		}
	}
	printf("\n");
	Logger::log(LogLevel::Info, "Done");

	//_deviceKeys.clearPrivateKeys();
}

void CudaKeySearchDevice::reGenerateStartingPoints()
{
	uint64_t totalPoints = (uint64_t)_pointsPerThread * _threads * _blocks;
	uint64_t totalMemory = totalPoints * 40;

	std::vector<secp256k1::uint256> exponents;

	//Logger::log(LogLevel::Info, "Regenerating " + util::formatThousands(totalPoints) + " starting points (" + util::format("%.1f", (double)totalMemory / (double)(1024 * 1024)) + "MB)");

	// Generate key pairs for k, k+1, k+2 ... k + <total points in parallel - 1>
	secp256k1::uint256 privKey = _startExponent;

	exponents.push_back(privKey);

	for (uint64_t i = 1; i < totalPoints; i++) {
		privKey = privKey.add(_stride);
		exponents.push_back(privKey);
	}

	cudaCall(_deviceKeys.reInit(_blocks, _threads, _pointsPerThread, exponents));

	// Show progress in 10% increments
	double pct = 10.0;
	for (int i = 1; i <= 256; i++) {
		cudaCall(_deviceKeys.doStep());

		if (((double)i / 256.0) * 100.0 >= pct) {
			//if (pct <= 10.0)
			//	Logger::log2(LogLevel::Info, util::format("%.1f%%", pct));
			//else
			//	printf("  %.1f%%", pct);
			pct += 10.0;
		}
	}
	//printf("\n");
	//Logger::log(LogLevel::Info, "Done");

	//_deviceKeys.clearPrivateKeys();
}

void CudaKeySearchDevice::setTargets(const std::set<KeySearchTargetHash160>& targets)
{
	_targetsHash160.clear();

	for (std::set<KeySearchTargetHash160>::iterator i = targets.begin(); i != targets.end(); ++i) {
		hash160 h(i->value);
		_targetsHash160.push_back(h);
	}

	cudaCall(_targetLookup.setTargets(_targetsHash160));
}

void CudaKeySearchDevice::setTargets(const std::set<KeySearchTargetXPoint>& targets)
{
	_targetsXPoint.clear();

	for (std::set<KeySearchTargetXPoint>::iterator i = targets.begin(); i != targets.end(); ++i) {
		xpoint h(i->value);
		_targetsXPoint.push_back(h);
	}

	cudaCall(_targetLookup.setTargets(_targetsXPoint));
}


void CudaKeySearchDevice::doStep()
{
	uint64_t numKeys = (uint64_t)_blocks * _threads * _pointsPerThread;

	try {
		if (_iterations < 2 && _startExponent.cmp(numKeys) <= 0) {
			callKeyFinderKernel(_blocks, _threads, _pointsPerThread, true, _compression, _searchMode);
		}
		else {
			callKeyFinderKernel(_blocks, _threads, _pointsPerThread, false, _compression, _searchMode);
		}
	}
	catch (cuda::CudaException ex) {
		throw KeySearchException(ex.msg);
	}

	getResultsInternal();

	_iterations++;
}

uint64_t CudaKeySearchDevice::keysPerStep()
{
	return (uint64_t)_blocks * _threads * _pointsPerThread;
}

std::string CudaKeySearchDevice::getDeviceName()
{
	return _deviceName;
}

void CudaKeySearchDevice::getMemoryInfo(uint64_t& freeMem, uint64_t& totalMem)
{
	cudaCall(cudaMemGetInfo(&freeMem, &totalMem));
}

void CudaKeySearchDevice::removeTargetFromListHash160(const unsigned int hash[5])
{
	size_t count = _targetsHash160.size();

	while (count) {
		if (memcmp(hash, _targetsHash160[count - 1].h, 20) == 0) {
			_targetsHash160.erase(_targetsHash160.begin() + count - 1);
			return;
		}
		count--;
	}
}

void CudaKeySearchDevice::removeTargetFromListXPoint(const unsigned int point[8])
{
	size_t count = _targetsXPoint.size();

	while (count) {
		if (memcmp(point, _targetsXPoint[count - 1].p, 32) == 0) {
			_targetsXPoint.erase(_targetsXPoint.begin() + count - 1);
			return;
		}
		count--;
	}
}

bool CudaKeySearchDevice::isTargetInListHash160(const unsigned int hash[5])
{
	size_t count = _targetsHash160.size();

	while (count) {
		if (memcmp(hash, _targetsHash160[count - 1].h, 20) == 0) {
			return true;
		}
		count--;
	}

	return false;
}

bool CudaKeySearchDevice::isTargetInListXPoint(const unsigned int point[8])
{
	size_t count = _targetsXPoint.size();

	while (count) {
		if (memcmp(point, _targetsXPoint[count - 1].p, 32) == 0) {
			return true;
		}
		count--;
	}

	return false;
}

uint32_t CudaKeySearchDevice::getPrivateKeyOffset(int thread, int block, int idx)
{
	// Total number of threads
	int totalThreads = _blocks * _threads;

	int base = idx * totalThreads;

	// Global ID of the current thread
	int threadId = block * _threads + thread;

	return base + threadId;
}

void CudaKeySearchDevice::getResultsInternal()
{
	int count = _resultList.size();
	int actualCount = 0;
	if (count == 0) {
		return;
	}

	unsigned char* ptr = new unsigned char[count * sizeof(CudaDeviceResult)];

	_resultList.read(ptr, count);

	if (_searchMode == SearchMode::ADDRESS) {

		for (int i = 0; i < count; i++) {
			struct CudaDeviceResult* rPtr = &((struct CudaDeviceResult*)ptr)[i];

			// might be false-positive
			if (!isTargetInListHash160(rPtr->digest)) {
				continue;
			}
			actualCount++;

			KeySearchResult minerResult;

			// Calculate the private key based on the number of iterations and the current thread
			secp256k1::uint256 offset = (secp256k1::uint256((uint64_t)_blocks * _threads * _pointsPerThread * _iterations) + secp256k1::uint256(getPrivateKeyOffset(rPtr->thread, rPtr->block, rPtr->idx))) * _stride;
			secp256k1::uint256 privateKey = secp256k1::addModN(_startExponent, offset);

			minerResult.privateKey = privateKey;
			minerResult.compressed = rPtr->compressed;

			memcpy(minerResult.hash, rPtr->digest, 20);

			minerResult.publicKey = secp256k1::ecpoint(secp256k1::uint256(rPtr->x, secp256k1::uint256::BigEndian), secp256k1::uint256(rPtr->y, secp256k1::uint256::BigEndian));

			removeTargetFromListHash160(rPtr->digest);

			_results.push_back(minerResult);
		}

		// Reload the bloom filters
		if (actualCount) {
			cudaCall(_targetLookup.setTargets(_targetsHash160));
		}
	}
	else if (_searchMode == SearchMode::XPOINT) {

		for (int i = 0; i < count; i++) {
			struct CudaDeviceResult* rPtr = &((struct CudaDeviceResult*)ptr)[i];

			// might be false-positive
			if (!isTargetInListXPoint(rPtr->x)) {
				continue;
			}
			actualCount++;

			KeySearchResult minerResult;

			// Calculate the private key based on the number of iterations and the current thread
			secp256k1::uint256 offset = (secp256k1::uint256((uint64_t)_blocks * _threads * _pointsPerThread * _iterations) + secp256k1::uint256(getPrivateKeyOffset(rPtr->thread, rPtr->block, rPtr->idx))) * _stride;
			secp256k1::uint256 privateKey = secp256k1::addModN(_startExponent, offset);

			minerResult.privateKey = privateKey;
			minerResult.compressed = rPtr->compressed;

			memcpy(minerResult.x, rPtr->x, 32);

			minerResult.publicKey = secp256k1::ecpoint(secp256k1::uint256(rPtr->x, secp256k1::uint256::BigEndian), secp256k1::uint256(rPtr->y, secp256k1::uint256::BigEndian));

			removeTargetFromListXPoint(rPtr->x);

			_results.push_back(minerResult);
		}

		// Reload the bloom filters
		if (actualCount) {
			cudaCall(_targetLookup.setTargets(_targetsXPoint));
		}
	}

	delete[] ptr;

	_resultList.clear();

}

/*
// Verify a private key produces the public key and hash
bool CudaKeySearchDevice::verifyKeyHash160(const secp256k1::uint256& privateKey, const secp256k1::ecpoint& publicKey, const unsigned int hash[5], bool compressed)
{
	secp256k1::ecpoint g = secp256k1::G();

	secp256k1::ecpoint p = secp256k1::multiplyPoint(privateKey, g);

	if (!(p == publicKey)) {
		return false;
	}

	unsigned int xWords[8];
	unsigned int yWords[8];

	p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
	p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

	unsigned int digest[5];
	if (compressed) {
		Hash::hashPublicKeyCompressed(xWords, yWords, digest);
	}
	else {
		Hash::hashPublicKey(xWords, yWords, digest);
	}

	for (int i = 0; i < 5; i++) {
		if (digest[i] != hash[i]) {
			return false;
		}
	}

	return true;
}
*/

size_t CudaKeySearchDevice::getResults(std::vector<KeySearchResult>& resultsOut)
{
	for (int i = 0; i < _results.size(); i++) {
		resultsOut.push_back(_results[i]);
	}
	_results.clear();

	return resultsOut.size();
}

secp256k1::uint256 CudaKeySearchDevice::getNextKey()
{
	uint64_t totalPoints = (uint64_t)_pointsPerThread * _threads * _blocks;

	return _startExponent + secp256k1::uint256(totalPoints) * _iterations * _stride;
}

void CudaKeySearchDevice::updateStride(const secp256k1::uint256& stride)
{
	_stride = stride;

	// regenerate starting points
	reGenerateStartingPoints();

	// Set the new incrementor
	secp256k1::ecpoint g = secp256k1::G();
	secp256k1::ecpoint p = secp256k1::multiplyPoint(secp256k1::uint256((uint64_t)_threads * _blocks * _pointsPerThread) * _stride, g);

	cudaCall(setIncrementorPoint(p.x, p.y));

	_iterations = 0;
}