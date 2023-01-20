#include "util/monad/error.h"

#include <iostream>

glow::util::monad::Error::Error(const std::string &message) : message(message) {}

const std::string &glow::util::monad::Error::getMessage() const {
  return message;
}

std::shared_ptr<glow::util::monad::Error> glow::util::monad::Error::joinErrors(const std::vector<std::shared_ptr<Error>> &errors) {
  size_t totalLength = 0;
  for (const auto &err : errors) {
    totalLength += err->message.size() + 1; // Plus 1 for newline separator
  }

  std::string combinedMessage;
  combinedMessage.reserve(totalLength);

  for (const auto &err : errors) {
    combinedMessage.append(err->message);
    combinedMessage.append("\n");
  }

  return std::make_shared<glow::util::monad::Error>(combinedMessage);
}