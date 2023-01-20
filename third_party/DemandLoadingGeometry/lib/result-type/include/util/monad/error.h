#pragma once

#include <memory>
#include <string>
#include <vector>

namespace glow::util::monad {
class Error {
public:
  Error(){}; // FIXME remove
  Error(const std::string &message);
  const std::string &getMessage() const;
  static std::shared_ptr<Error> joinErrors(const std::vector<std::shared_ptr<Error>> &errors);

private:
  std::string message;
};
} // namespace glow::util::monad

using Err = std::shared_ptr<glow::util::monad::Error>;
