#ifndef FIXEDSIZEDEQUE
#define FIXEDSIZEDEQUE
#include <deque>

template <typename T>
struct FixedSizeDeque {
    FixedSizeDeque(int s) : capacity(s) {}
    void addToHistory(const T& t) {
        x.push_front(t);
        if (x.size() > capacity) x.pop_back();
    }
    const T& operator[] (int i) const { return x.at(i); }
    int size() const { return x.size(); }
    int capacity;
private:
    std::deque<T> x;
};
#endif /* FIXEDSIZEDEQUE */
