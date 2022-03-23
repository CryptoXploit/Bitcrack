#ifndef _KEY_FINDER_H
#define _KEY_FINDER_H

#include <stdint.h>
#include <vector>
#include <set>
#include "../Secp256k1/secp256k1.h"
#include "KeySearchTypes.h"
#include "KeySearchDevice.h"


class KeyFinder {

private:

	KeySearchDevice* _device;

	unsigned int _compression;
	unsigned int _searchMode;

	std::set<KeySearchTargetHash160> _targetsHash160;
	std::set<KeySearchTargetXPoint> _targetsXPoint;

	uint64_t _statusInterval;

	secp256k1::uint256 _stride = 1;
	uint64_t _iterCount;
	uint64_t _total;
	uint64_t _totalTime;

	secp256k1::uint256 _startKey;
	secp256k1::uint256 _endKey;

	// Each index of each thread gets a flag to indicate if it found a valid hash
	bool _running;

	bool _randomStride;
	bool _continueAfterEnd;
	uint32_t _randomSrtrideBits;
	uint32_t _rStrideCount;
	std::vector<uint32_t> _rStrideHistory;

	void(*_resultCallback)(KeySearchResult);
	void(*_statusCallback)(KeySearchStatus);


	static void defaultResultCallback(KeySearchResult result);
	static void defaultStatusCallback(KeySearchStatus status);

	void removeTargetFromListHash160(const unsigned int value[5]);
	void removeTargetFromListXPoint(const unsigned int value[8]);
	bool isTargetInListHash160(const unsigned int value[5]);
	bool isTargetInListXPoint(const unsigned int value[8]);
	//void setTargetsOnDevice();


	void reSetupEverything();

public:

	KeyFinder(const secp256k1::uint256& startKey, const secp256k1::uint256& endKey, int compression, int searchMode,
		KeySearchDevice* device, const secp256k1::uint256& stride, bool randomStride, bool continueAfterEnd, 
		uint32_t randomSrtrideBits);

	~KeyFinder();

	void init();
	void run();
	void stop();

	void setResultCallback(void(*callback)(KeySearchResult));
	void setStatusCallback(void(*callback)(KeySearchStatus));
	void setStatusInterval(uint64_t interval);

	void setTargets(std::string targetFile);
	void setTargets(std::vector<std::string>& targets);

	secp256k1::uint256 getNextKey();


	static uint64_t LZC(uint64_t x);
	static uint64_t TZC(uint64_t x);
};

#endif