#include <mutex>

#include <debug/profile.h>

namespace ir {

#ifdef IR_DEBUG_PROFILE

std::unique_ptr<Profile> Profile::profile_ = nullptr;

Profile &Profile::getInstance() {
    if (profile_ == nullptr) {
        profile_.reset(new Profile());
    }
    return *profile_;
}

Profile::Profile() : os_("debug_profile.csv") {
    os_ << "Name,Detail,Time(ms)" << std::endl;
}

void Profile::add(const std::string &name, const std::string &detail,
                  double t) {
    os_ << "\"" << name << "\",\"" << detail << "\"," << t << std::endl;
}

ProfileGuard::ProfileGuard(const std::string &name, const std::string &detail)
    : name_(name), detail_(detail),
      begin_(std::chrono::high_resolution_clock::now()) {}

ProfileGuard::~ProfileGuard() {
    namespace ch = std::chrono;
    auto end = ch::high_resolution_clock::now();
    double t = ch::duration_cast<ch::duration<double>>(end - begin_).count() *
               1000; // ms
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    Profile::getInstance().add(name_, detail_, t);
}

#endif // IR_DEBUG_PROFILE

} // namespace ir
