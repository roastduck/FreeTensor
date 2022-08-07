#ifndef FREE_TENSOR_LOAD_DRIVER_H
#define FREE_TENSOR_LOAD_DRIVER_H

#include <driver/array.h>
#include <driver/device.h>
#include <driver/target.h>
#include <ref.h>
namespace freetensor {

Ref<Target> loadTarget(const std::string &txt);
Ref<Device> loadDevice(const std::string &txt);
Ref<Array> loadArray(const std::pair<const std::string &, const std::string &>);

/**
 * The function is only for serialization
 * The original Array constructor is disabled
 * See include/driver/array.h for more info
 */
Ref<Array> newArray(const std::vector<size_t> &shape_,
                    const std::string &dtype_, const std::string &data_);

} // namespace freetensor

#endif // FREE_TENSOR_LOAD_DRIVER_H
