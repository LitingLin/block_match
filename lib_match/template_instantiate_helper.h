#pragma once

#define RuntimeTypeInference(type, exp) \
	if (type == typeid(bool)) \
		exp(bool); \
	else if (type == typeid(uint8_t)) \
		exp(uint8_t); \
	else if (type == typeid(int8_t)) \
		exp(int8_t); \
	else if (type == typeid(uint16_t)) \
		exp(uint16_t); \
	else if (type == typeid(int16_t)) \
		exp(int16_t); \
	else if (type == typeid(uint32_t)) \
		exp(uint32_t); \
	else if (type == typeid(int32_t)) \
		exp(int32_t); \
	else if (type == typeid(uint64_t)) \
		exp(uint64_t); \
	else if (type == typeid(int64_t)) \
		exp(int64_t); \
	else if (type == typeid(float)) \
		exp(float); \
	else if (type == typeid(double)) \
		exp(double); \
	else \
		NOT_IMPLEMENTED_ERROR

#define RuntimeTypeInference2(type1, type2, exp) \
	if (type1 == typeid(bool)) { \
		if (type2 == typeid(bool)) \
			exp(bool, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(bool, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(bool, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(bool, uint16_t); \
		else if (type2 == typeid(int16_t)) \
			exp(bool, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(bool, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(bool, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(bool, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(bool, int64_t); \
		else if (type2 == typeid(float)) \
			exp(bool, float); \
		else if (type2 == typeid(double)) \
			exp(bool, double); } \
	else if (type1 == typeid(uint8_t)) { \
		if (type2 == typeid(bool)) \
			exp(uint8_t, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(uint8_t, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(uint8_t, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(uint8_t, uint16_t); \
		else if (type2 == typeid(int16_t)) \
			exp(uint8_t, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(uint8_t, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(uint8_t, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(uint8_t, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(uint8_t, int64_t); \
		else if (type2 == typeid(float)) \
			exp(uint8_t, float); \
		else if (type2 == typeid(double)) \
			exp(uint8_t, double); } \
	else if (type1 == typeid(int8_t)) { \
		if (type2 == typeid(bool)) \
			exp(int8_t, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(int8_t, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(int8_t, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(int8_t, uint16_t); \
		else if (type2 == typeid(int16_t)) \
			exp(int8_t, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(int8_t, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(int8_t, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(int8_t, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(int8_t, int64_t); \
		else if (type2 == typeid(float)) \
			exp(int8_t, float); \
		else if (type2 == typeid(double)) \
			exp(int8_t, double); } \
	else if (type1 == typeid(uint16_t)) { \
		if (type2 == typeid(bool)) \
			exp(uint16_t, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(uint16_t, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(uint16_t, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(uint16_t, uint16_t); \
		else if (type2 == typeid(int16_t)) \
			exp(uint16_t, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(uint16_t, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(uint16_t, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(uint16_t, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(uint16_t, int64_t); \
		else if (type2 == typeid(float)) \
			exp(uint16_t, float); \
		else if (type2 == typeid(double)) \
			exp(uint16_t, double); } \
	else if (type1 == typeid(int16_t)) { \
		if (type2 == typeid(bool)) \
			exp(int16_t, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(int16_t, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(int16_t, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(int16_t, uint16_t); \
		else if (type2 == typeid(int16_t)) \
			exp(int16_t, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(int16_t, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(int16_t, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(int16_t, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(int16_t, int64_t); \
		else if (type2 == typeid(float)) \
			exp(int16_t, float); \
		else if (type2 == typeid(double)) \
			exp(int16_t, double); } \
	else if (type1 == typeid(uint32_t)) { \
		if (type2 == typeid(bool)) \
			exp(uint32_t, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(uint32_t, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(uint32_t, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(uint32_t, uint16_t); \
		else if (type2 == typeid(int16_t)) \
			exp(uint32_t, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(uint32_t, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(uint32_t, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(uint32_t, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(uint32_t, int64_t); \
		else if (type2 == typeid(float)) \
			exp(uint32_t, float); \
		else if (type2 == typeid(double)) \
			exp(uint32_t, double); } \
	else if (type1 == typeid(int32_t)) { \
		if (type2 == typeid(bool)) \
			exp(int32_t, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(int32_t, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(int32_t, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(int32_t, uint16_t); \
		else if (type2 == typeid(int16_t)) \
			exp(int32_t, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(int32_t, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(int32_t, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(int32_t, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(int32_t, int64_t); \
		else if (type2 == typeid(float)) \
			exp(int32_t, float); \
		else if (type2 == typeid(double)) \
			exp(int32_t, double); } \
	else if (type1 == typeid(uint64_t)) { \
		if (type2 == typeid(bool)) \
			exp(uint64_t, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(uint64_t, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(uint64_t, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(uint64_t, uint16_t); \
		else if (type2 == typeid(int16_t)) \
			exp(uint64_t, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(uint64_t, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(uint64_t, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(uint64_t, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(uint64_t, int64_t); \
		else if (type2 == typeid(float)) \
			exp(uint64_t, float); \
		else if (type2 == typeid(double)) \
			exp(uint64_t, double); } \
	else if (type1 == typeid(int64_t)) { \
		if (type2 == typeid(bool)) \
			exp(int64_t, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(int64_t, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(int64_t, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(int64_t, uint16_t); \
		else if (type2 == typeid(int16_t)) \
			exp(int64_t, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(int64_t, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(int64_t, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(int64_t, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(int64_t, int64_t); \
		else if (type2 == typeid(float)) \
			exp(int64_t, float); \
		else if (type2 == typeid(double)) \
			exp(int64_t, double); } \
	else if (type1 == typeid(float)) { \
		if (type2 == typeid(bool)) \
			exp(float, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(float, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(float, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(float, uint16_t); \
			else if (type2 == typeid(int16_t)) \
			exp(float, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(float, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(float, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(float, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(float, int64_t); \
		else if (type2 == typeid(float)) \
			exp(float, float); \
		else if (type2 == typeid(double)) \
			exp(float, double); } \
	else if (type1 == typeid(double)) { \
		if (type2 == typeid(bool)) \
			exp(double, bool); \
		else if (type2 == typeid(uint8_t)) \
			exp(double, uint8_t); \
		else if (type2 == typeid(int8_t)) \
			exp(double, int8_t); \
		else if (type2 == typeid(uint16_t)) \
			exp(double, uint16_t); \
		else if (type2 == typeid(int16_t)) \
			exp(double, int16_t); \
		else if (type2 == typeid(uint32_t)) \
			exp(double, uint32_t); \
		else if (type2 == typeid(int32_t)) \
			exp(double, int32_t); \
		else if (type2 == typeid(uint64_t)) \
			exp(double, uint64_t); \
		else if (type2 == typeid(int64_t)) \
			exp(double, int64_t); \
		else if (type2 == typeid(float)) \
			exp(double, float); \
		else if (type2 == typeid(double)) \
			exp(double, double); } \
	else \
		NOT_IMPLEMENTED_ERROR

#define InstantiateTemplateFloating(exp) \
	exp(float); \
	exp(double)


#define InstantiateTemplate(exp) \
	exp(bool); \
	exp(uint8_t); \
	exp(int8_t); \
	exp(uint16_t); \
	exp(int16_t); \
	exp(uint32_t); \
	exp(int32_t); \
	exp(uint64_t); \
	exp(int64_t); \
	exp(float); \
	exp(double)

#define InstantiateTemplate2(exp) \
	exp(bool, bool); \
	exp(bool, uint8_t); \
	exp(bool, int8_t); \
	exp(bool, uint16_t); \
	exp(bool, int16_t); \
	exp(bool, uint32_t); \
	exp(bool, int32_t); \
	exp(bool, uint64_t); \
	exp(bool, int64_t); \
	exp(bool, float); \
	exp(bool, double); \
	exp(uint8_t, bool); \
	exp(uint8_t, uint8_t); \
	exp(uint8_t, int8_t); \
	exp(uint8_t, uint16_t); \
	exp(uint8_t, int16_t); \
	exp(uint8_t, uint32_t); \
	exp(uint8_t, int32_t); \
	exp(uint8_t, uint64_t); \
	exp(uint8_t, int64_t); \
	exp(uint8_t, float); \
	exp(uint8_t, double); \
	exp(int8_t, bool); \
	exp(int8_t, uint8_t); \
	exp(int8_t, int8_t); \
	exp(int8_t, uint16_t); \
	exp(int8_t, int16_t); \
	exp(int8_t, uint32_t); \
	exp(int8_t, int32_t); \
	exp(int8_t, uint64_t); \
	exp(int8_t, int64_t); \
	exp(int8_t, float); \
	exp(int8_t, double); \
	exp(uint16_t, bool); \
	exp(uint16_t, uint8_t); \
	exp(uint16_t, int8_t); \
	exp(uint16_t, uint16_t); \
	exp(uint16_t, int16_t); \
	exp(uint16_t, uint32_t); \
	exp(uint16_t, int32_t); \
	exp(uint16_t, uint64_t); \
	exp(uint16_t, int64_t); \
	exp(uint16_t, float); \
	exp(uint16_t, double); \
	exp(int16_t, bool); \
	exp(int16_t, uint8_t); \
	exp(int16_t, int8_t); \
	exp(int16_t, uint16_t); \
	exp(int16_t, int16_t); \
	exp(int16_t, uint32_t); \
	exp(int16_t, int32_t); \
	exp(int16_t, uint64_t); \
	exp(int16_t, int64_t); \
	exp(int16_t, float); \
	exp(int16_t, double); \
	exp(uint32_t, bool); \
	exp(uint32_t, uint8_t); \
	exp(uint32_t, int8_t); \
	exp(uint32_t, uint16_t); \
	exp(uint32_t, int16_t); \
	exp(uint32_t, uint32_t); \
	exp(uint32_t, int32_t); \
	exp(uint32_t, uint64_t); \
	exp(uint32_t, int64_t); \
	exp(uint32_t, float); \
	exp(uint32_t, double); \
	exp(int32_t, bool); \
	exp(int32_t, uint8_t); \
	exp(int32_t, int8_t); \
	exp(int32_t, uint16_t); \
	exp(int32_t, int16_t); \
	exp(int32_t, uint32_t); \
	exp(int32_t, int32_t); \
	exp(int32_t, uint64_t); \
	exp(int32_t, int64_t); \
	exp(int32_t, float); \
	exp(int32_t, double); \
	exp(uint64_t, bool); \
	exp(uint64_t, uint8_t); \
	exp(uint64_t, int8_t); \
	exp(uint64_t, uint16_t); \
	exp(uint64_t, int16_t); \
	exp(uint64_t, uint32_t); \
	exp(uint64_t, int32_t); \
	exp(uint64_t, uint64_t); \
	exp(uint64_t, int64_t); \
	exp(uint64_t, float); \
	exp(uint64_t, double); \
	exp(int64_t, bool); \
	exp(int64_t, uint8_t); \
	exp(int64_t, int8_t); \
	exp(int64_t, uint16_t); \
	exp(int64_t, int16_t); \
	exp(int64_t, uint32_t); \
	exp(int64_t, int32_t); \
	exp(int64_t, uint64_t); \
	exp(int64_t, int64_t); \
	exp(int64_t, float); \
	exp(int64_t, double); \
	exp(float, bool); \
	exp(float, uint8_t); \
	exp(float, int8_t); \
	exp(float, uint16_t); \
	exp(float, int16_t); \
	exp(float, uint32_t); \
	exp(float, int32_t); \
	exp(float, uint64_t); \
	exp(float, int64_t); \
	exp(float, float); \
	exp(float, double); \
	exp(double, bool); \
	exp(double, uint8_t); \
	exp(double, int8_t); \
	exp(double, uint16_t); \
	exp(double, int16_t); \
	exp(double, uint32_t); \
	exp(double, int32_t); \
	exp(double, uint64_t); \
	exp(double, int64_t); \
	exp(double, float); \
	exp(double, double)