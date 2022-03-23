#ifndef _KEY_FINDER_TYPES
#define _KEY_FINDER_TYPES

#include<stdint.h>
#include<string>
#include "../Secp256k1/secp256k1.h"

namespace PointCompressionType {
	enum Value {
		COMPRESSED = 0,
		UNCOMPRESSED = 1,
		BOTH = 2
	};
}

namespace SearchMode {
	enum Value {
		ADDRESS = 0,
		XPOINT = 1,
	};
}

typedef struct hash160 {

	unsigned int h[5];

	hash160(const unsigned int hash[5])
	{
		memcpy(h, hash, sizeof(unsigned int) * 5);
	}
}hash160;

typedef struct xpoint {

	unsigned int p[8];

	xpoint(const unsigned int x[8])
	{
		memcpy(p, x, sizeof(unsigned int) * 8);
	}
}xpoint;

typedef struct {
	int device;
	double speed;
	uint64_t total;
	uint64_t totalTime;
	std::string deviceName;
	uint64_t freeMemory;
	uint64_t deviceMemory;
	uint64_t targets;
	secp256k1::uint256 nextKey;
	secp256k1::uint256 stride;
	uint32_t rStrideCount;
}KeySearchStatus;


class KeySearchTargetHash160 {

public:
	unsigned int value[5];

	KeySearchTargetHash160()
	{
		memset(value, 0, sizeof(value));
	}

	KeySearchTargetHash160(const unsigned int h[5])
	{
		for (int i = 0; i < 5; i++) {
			value[i] = h[i];
		}
	}


	bool operator==(const KeySearchTargetHash160& t) const
	{
		for (int i = 0; i < 5; i++) {
			if (value[i] != t.value[i]) {
				return false;
			}
		}

		return true;
	}

	bool operator<(const KeySearchTargetHash160& t) const
	{
		for (int i = 0; i < 5; i++) {
			if (value[i] < t.value[i]) {
				return true;
			}
			else if (value[i] > t.value[i]) {
				return false;
			}
		}

		return false;
	}

	bool operator>(const KeySearchTargetHash160& t) const
	{
		for (int i = 0; i < 5; i++) {
			if (value[i] > t.value[i]) {
				return true;
			}
			else if (value[i] < t.value[i]) {
				return false;
			}
		}

		return false;
	}
};



class KeySearchTargetXPoint {

public:
	unsigned int value[8];

	KeySearchTargetXPoint()
	{
		memset(value, 0, sizeof(value));
	}

	KeySearchTargetXPoint(const unsigned int h[8])
	{
		for (int i = 0; i < 8; i++) {
			value[i] = h[i];
		}
	}


	bool operator==(const KeySearchTargetXPoint& t) const
	{
		for (int i = 0; i < 8; i++) {
			if (value[i] != t.value[i]) {
				return false;
			}
		}

		return true;
	}

	bool operator<(const KeySearchTargetXPoint& t) const
	{
		for (int i = 0; i < 8; i++) {
			if (value[i] < t.value[i]) {
				return true;
			}
			else if (value[i] > t.value[i]) {
				return false;
			}
		}

		return false;
	}

	bool operator>(const KeySearchTargetXPoint& t) const
	{
		for (int i = 0; i < 8; i++) {
			if (value[i] > t.value[i]) {
				return true;
			}
			else if (value[i] < t.value[i]) {
				return false;
			}
		}

		return false;
	}
};


#endif