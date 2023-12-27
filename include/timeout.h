#ifndef FREETENSOR_TIMEOUT_H
#define FREETENSOR_TIMEOUT_H

#include <functional>
#include <vector>

namespace freetensor {

/**
 * Run a function with timeout
 *
 * If timed out, the internal state of the function is left untouched, so there
 * will be resource leak, and it should be tolerated.
 *
 * NOTE: There are other approaches without resource leak but still not
 * preferred: `fork` + `kill` with pages copied on write, or `fork` + `exec` a
 * stand-alone executable + `kill` are possible, but both come with too much
 * overhead. Waiting a thread until timeout and left it running is a waste of
 * CPU time.
 *
 * @return : True if the function successfully returns. False if the time is out
 */
bool timeout(const std::function<void()> &func, int seconds);

} // namespace freetensor

#endif // FREETENSOR_TIMEOUT_H
