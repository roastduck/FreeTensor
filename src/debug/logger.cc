#include <debug/logger.h>

namespace ir {

LogCtrl LogCtrl::instance_;
std::mutex Logger::lock_;

} // namespace ir
