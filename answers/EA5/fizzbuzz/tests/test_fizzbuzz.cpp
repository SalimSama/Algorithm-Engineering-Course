#include <catch2/catch_test_macros.hpp>
#include "fizzbuzz.h"

TEST_CASE("FizzBuzz computes the correct sequence", "[fizzbuzz]") {
    int n = 20;
    auto result = fizzbuzz(n);

    REQUIRE(result.size() == n);

    std::vector<std::string> expected = {
        "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
        "11", "Fizz", "13", "14", "FizzBuzz", "16", "17", "Fizz", "19", "Buzz"
    };

    REQUIRE(result == expected);
}

TEST_CASE("Test positives", "[classic]")
{
   SECTION("Test all up to 10") {
      REQUIRE(fizzbuzz_single(1) == "1"); 
      REQUIRE(fizzbuzz_single(2) == "2");
      REQUIRE(fizzbuzz_single(3) == "Fizz");
      REQUIRE(fizzbuzz_single(4) == "4");
      REQUIRE(fizzbuzz_single(5) == "Buzz");
      REQUIRE(fizzbuzz_single(6) == "Fizz");
      REQUIRE(fizzbuzz_single(7) == "7");
      REQUIRE(fizzbuzz_single(8) == "8");
      REQUIRE(fizzbuzz_single(9) == "Fizz");
      REQUIRE(fizzbuzz_single(10) == "Buzz");
   }

   SECTION("Test all multiples of 3 only up to 100") {
      for (int i = 3; i <= 100; i += 3) {
         if (i % 5 != 0) REQUIRE(fizzbuzz_single(i) == "Fizz");
      }
   }

   SECTION("Test all multiples of 5 only up to 100") {
      for (int i = 5; i <= 100; i += 5) {
         if (i % 3 != 0) REQUIRE(fizzbuzz_single(i) == "Buzz");
      }
   }

   SECTION("Test all multiples of 3 and 5 up to 100") {
      for (int i = 15; i <= 100; i += 15) {
         REQUIRE(fizzbuzz_single(i) == "FizzBuzz");
      }
   }
}

TEST_CASE("Test negatives", "[classic]")
{
   REQUIRE(fizzbuzz_single(-1) == "-1");
   REQUIRE(fizzbuzz_single(-2) == "-2");
   REQUIRE(fizzbuzz_single(-3) == "Fizz");
   REQUIRE(fizzbuzz_single(-4) == "-4");
   REQUIRE(fizzbuzz_single(-5) == "Buzz");
   REQUIRE(fizzbuzz_single(-6) == "Fizz");
   REQUIRE(fizzbuzz_single(-7) == "-7");
   REQUIRE(fizzbuzz_single(-8) == "-8");
   REQUIRE(fizzbuzz_single(-9) == "Fizz");
   REQUIRE(fizzbuzz_single(-10) == "Buzz");
}
