#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <string>

//#define DATA_IN(path) TEST_DATA_PATH "/" path
//#define DATA_OUT(path) "out/" path

std::string DATA_IN(const std::string& path)
{
    return std::string(TEST_DATA_PATH) + "/" + path;
}

std::string DATA_OUT(const std::string& path)
{
    return "out/" + path;
}

#endif
