#ifndef FREE_TENSOR_PROFILE_H
#define FREE_TENSOR_PROFILE_H

#include <chrono>
#include <fstream>
#include <memory>

namespace freetensor {

#ifdef FT_DEBUG_PROFILE

class Profile {
    std::ofstream os_;

    static std::unique_ptr<Profile> profile_;

  public:
    Profile();

    void add(const std::string &name, const std::string &detail, double t);

    static Profile &getInstance();
};

class ProfileGuard {
    std::string name_, detail_;
    std::chrono::time_point<std::chrono::high_resolution_clock> begin_;

  public:
    ProfileGuard(const std::string &name, const std::string &detail);
    ~ProfileGuard();
};

#define DEBUG_PROFILE(name)                                                    \
    ProfileGuard __profGuard(name, __FILE__ ":" + std::to_string(__LINE__))

#define DEBUG_PROFILE_VERBOSE(name, detail)                                    \
    ProfileGuard __profGuard(name, __FILE__ ":" + std::to_string(__LINE__) +   \
                                       ": " + detail)

#else // FT_DEBUG_PROFILE

#define DEBUG_PROFILE(name)

#define DEBUG_PROFILE_VERBOSE(name, detail)

#endif // FT_DEBUG_PROFILE

} // namespace freetensor

#endif // FREE_TENSOR_PROFILE_H
