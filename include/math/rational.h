#ifndef FREE_TENSOR_RATIONAL_H
#define FREE_TENSOR_RATIONAL_H

#include <iostream>
#include <numeric>

namespace freetensor {

template <class T> struct Rational {
    T p_, q_; // p_ / q_

    Rational(T p = 0, T q = 1) : p_(p), q_(q) {
        if (p == 0) {
            q_ = 1;
        } else {
            T g = std::gcd(p, q);
            p_ /= g, q_ /= g;
            if (q_ < 0) {
                p_ = -p_, q_ = -q_;
            }
        }
    }

    friend bool operator==(const Rational<T> &lhs, const Rational<T> &rhs) {
        return lhs.p_ == rhs.p_ && lhs.q_ == rhs.q_;
    }

    friend Rational operator+(const Rational<T> &lhs, const Rational<T> &rhs) {
        T g = std::gcd(lhs.q_, rhs.q_);
        T p = rhs.q_ / g * lhs.p_ + lhs.q_ / g * rhs.p_;
        T q = lhs.q_ / g * rhs.q_;
        return Rational<T>{p, q};
    }

    friend Rational operator-(const Rational<T> &lhs, const Rational<T> &rhs) {
        T g = std::gcd(lhs.q_, rhs.q_);
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

    friend auto operator<=>(const Rational<T> &lhs, const Rational<T> &rhs) {
        T g = std::gcd(lhs.q_, rhs.q_);
        return rhs.q_ / g * lhs.p_ <=> lhs.q_ / g * rhs.p_;
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

} // namespace freetensor

namespace std {

template <class T>
freetensor::Rational<T> abs(const freetensor::Rational<T> &x) {
    return x < 0 ? -x : x;
}

} // namespace std

#endif // FREE_TENSOR_RATIONAL_H
