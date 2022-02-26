#ifndef RATIONAL_H
#define RATIONAL_H

#include <iostream>

#include <math/utils.h>

namespace ir {

template <class T> struct Rational {
    T p_, q_; // p_ / q_

    Rational(T p = 0, T q = 1) : p_(p), q_(q) {
        if (p == 0) {
            q_ = 1;
        } else {
            T g = gcd(p, q);
            p_ /= g, q_ /= g;
            if (q_ < 0) {
                p_ = -p_, q_ = -q_;
            }
        }
    }

    friend bool operator==(const Rational<T> &lhs, const Rational<T> &rhs) {
        return lhs.p_ == rhs.p_ && lhs.q_ == rhs.q_;
    }

    friend bool operator!=(const Rational<T> &lhs, const Rational<T> &rhs) {
        return lhs.p_ != rhs.p_ || lhs.q_ != rhs.q_;
    }

    friend Rational operator+(const Rational<T> &lhs, const Rational<T> &rhs) {
        T g = gcd(lhs.q_, rhs.q_);
        T p = rhs.q_ / g * lhs.p_ + lhs.q_ / g * rhs.p_;
        T q = lhs.q_ / g * rhs.q_;
        return Rational<T>{p, q};
    }

    friend Rational operator-(const Rational<T> &lhs, const Rational<T> &rhs) {
        T g = gcd(lhs.q_, rhs.q_);
        T p = rhs.q_ / g * lhs.p_ - lhs.q_ / g * rhs.p_;
        T q = lhs.q_ / g * rhs.q_;
        return Rational<T>{p, q};
    }

    friend Rational operator*(const Rational<T> &lhs, const Rational<T> &rhs) {
        T p = lhs.p_ * rhs.p_;
        T q = lhs.q_ * rhs.q_;
        return Rational<T>{p, q};
    }

    friend Rational operator/(const Rational<T> &lhs, const Rational<T> &rhs) {
        T p = lhs.p_ * rhs.q_;
        T q = lhs.q_ * rhs.p_;
        return Rational<T>{p, q};
    }

    friend std::ostream &operator<<(std::ostream &os, const Rational<T> &r) {
        os << "(" << r.p_ << " / " << r.q_ << ")";
        return os;
    }

    friend bool operator<(const Rational<T> &lhs, const Rational<T> &rhs) {
        T g = gcd(lhs.q_, rhs.q_);
        return rhs.q_ / g * lhs.p_ < lhs.q_ / g * rhs.p_;
    }

    friend bool operator>(const Rational<T> &lhs, const Rational<T> &rhs) {
        T g = gcd(lhs.q_, rhs.q_);
        return rhs.q_ / g * lhs.p_ > lhs.q_ / g * rhs.p_;
    }

    friend bool operator<=(const Rational<T> &lhs, const Rational<T> &rhs) {
        return !(lhs > rhs);
    }

    friend bool operator>=(const Rational<T> &lhs, const Rational<T> &rhs) {
        return !(lhs < rhs);
    }

    Rational &operator+=(const Rational<T> &other) {
        return *this = *this + other;
    }
    Rational &operator-=(const Rational<T> &other) {
        return *this = *this - other;
    }
    Rational &operator*=(const Rational<T> &other) {
        return *this = *this * other;
    }
    Rational &operator/=(const Rational<T> &other) {
        return *this = *this / other;
    }

    Rational operator-() const { return Rational{-p_, q_}; }
};

} // namespace ir

namespace std {

template <class T> ir::Rational<T> abs(const ir::Rational<T> &x) {
    return x < 0 ? -x : x;
}

} // namespace std

#endif // RATIONAL_H
