#include <result.hpp>

using namespace cpp;

#define UNWRAP(unwrappedValueName, resultValue)       \
  const auto __##unwrappedValueName = (resultValue);  \
  if ((__##unwrappedValueName).has_error()) {         \
    return failure((__##unwrappedValueName).error()); \
  };                                                  \
  const auto unwrappedValueName = (__##unwrappedValueName).value();

#define UNWRAP_OR_EXIT(unwrappedValueName, resultValue)       \
  const auto __##unwrappedValueName = (resultValue);          \
  if ((__##unwrappedValueName).has_error()) {                 \
    std::cerr << __##unwrappedValueName.error() << std::endl; \
    std::exit(1);                                             \
  }                                                           \
  const auto unwrappedValueName = (__##unwrappedValueName).value();

#define UNWRAP_MUT(unwrappedValueName, resultValue)   \
  auto __##unwrappedValueName = (resultValue);        \
  if ((__##unwrappedValueName).has_error()) {         \
    return failure((__##unwrappedValueName).error()); \
  };                                                  \
  auto unwrappedValueName = (__##unwrappedValueName).value();

#define UNWRAP_ASSIGN(unwrappedValueName, resultValue) \
  {                                                    \
    const auto __result = (resultValue);               \
    if ((__result).has_error()) {                      \
      return failure((__result).error());              \
    };                                                 \
    unwrappedValueName = (__result).value();           \
  }

#define UNWRAP_VOID(resultValue)         \
  {                                      \
    const auto __result = (resultValue); \
    if (__result.has_error()) {          \
      return failure(__result.error());  \
    };                                   \
  }

#define UNWRAP_TEST(unwrappedValueName, resultValue)                                       \
  const auto __##unwrappedValueName = (resultValue);                                       \
  if ((__##unwrappedValueName).has_error()) {                                              \
    std::cerr << "Error: " << (__##unwrappedValueName).error()->getMessage() << std::endl; \
    FAIL();                                                                                \
  };                                                                                       \
  const auto unwrappedValueName = (__##unwrappedValueName).value();
