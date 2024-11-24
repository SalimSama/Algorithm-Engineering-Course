#include "fizzbuzz.h"

std::vector<std::string> fizzbuzz(int n) {
    std::vector<std::string> result;
    result.reserve(n); // Reserve space for efficiency

    for (int i = 1; i <= n; ++i) {
        result.push_back(fizzbuzz_single(i)); // Use the single value function
    }

    return result;
}

std::string fizzbuzz_single(int i) {
    if (i % 15 == 0)
        return "FizzBuzz";
    else if (i % 3 == 0)
        return "Fizz";
    else if (i % 5 == 0)
        return "Buzz";
    else
        return std::to_string(i);
}
