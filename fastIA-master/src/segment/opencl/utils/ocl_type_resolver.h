#ifndef OCL_TYPE_RESOLVER_H
#define OCL_TYPE_RESOLVER_H

#include <string>

template <typename TYPE>
struct TypeResolver
{
};

template <>
struct TypeResolver<unsigned int>
{
    typedef int type;
    static const std::string type_as_string;
};

template <>
struct TypeResolver<int>
{
    typedef int type;
    static const std::string type_as_string;
};

template <>
struct TypeResolver<unsigned char>
{
    typedef int type;
    static const std::string type_as_string;
};

template <>
struct TypeResolver<char>
{
    typedef int type;
    static const std::string type_as_string;
};

#endif // OCL_TYPE_RESOLVER_H
