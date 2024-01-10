#include <mutex>
#include <setjmp.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include <debug.h>
#include <except.h>
#include <timeout.h>

namespace freetensor {

static thread_local sigjmp_buf jmpbuf;

static void sigAlarm(int sig) { siglongjmp(jmpbuf, 1); }

static void lazyInitAlarmHandler() {
    // Signal handler is per-process
    static bool alarmRegistered = false;
    static std::mutex lock;
    std::lock_guard gurad(lock);
    if (!alarmRegistered) {
        struct sigaction sa = {};
        sa.sa_handler = sigAlarm;
        if (sigaction(SIGALRM, &sa, NULL) == -1) {
            ERROR("sigaction failed");
        }
        alarmRegistered = true;
    }
}

// Workaround.
// https://stackoverflow.com/questions/16826898/error-struct-sigevent-has-no-member-named-sigev-notify-thread-id
#define sigev_notify_thread_id _sigev_un._tid

bool timeout(const std::function<void()> &func, int seconds) {
    lazyInitAlarmHandler();
    if (sigsetjmp(jmpbuf, 1) == 0) {
        // We use timer_create and timer_settime instead of alarm, because the
        // former supports per-thread notification
        timer_t timerid;
        struct sigevent sev = {};
        sev.sigev_notify = SIGEV_THREAD_ID;
        sev.sigev_signo = SIGALRM;
        sev.sigev_notify_thread_id = gettid();
        if (timer_create(CLOCK_THREAD_CPUTIME_ID, &sev, &timerid) == -1) {
            ERROR("timer_create failed");
        }
        struct itimerspec its = {};
        its.it_value.tv_sec = seconds;
        if (timer_settime(timerid, 0, &its, NULL) == -1) {
            ERROR("timer_settime failed");
        }

        try {
            func();
        } catch (...) {
            its = {}; // Cancel the timer
            if (timer_settime(timerid, 0, &its, NULL) == -1) {
                ERROR("timer_settime failed");
            }
            throw;
        }
        its = {}; // Cancel the timer
        if (timer_settime(timerid, 0, &its, NULL) == -1) {
            ERROR("timer_settime failed");
        }
        return true;
    } else {
        WARNING(FT_MSG << "Function timed out after " << seconds << " seconds");
        return false;
    }
}

} // namespace freetensor
