#include <cassert>
#ifdef WITH_TESTS
#include <gtest/gtest.h>
#endif
#include <iostream>
#include <vector>
#include <cstdlib>
#include <numeric>

auto linear_search(const std::vector<int>& vals, int key) {
  for (const auto& v : vals) {
    if (v == key) {
      return true;
    }
  }
  return false;
}

auto binary_search(const std::vector<int>& a, int key) {
  // Ensure our cast below is safe
  assert(a.size() < std::numeric_limits<int>::max());

  auto low = 0;
  auto high = static_cast<int>(a.size()) - 1;
  while (low <= high) {
    const auto mid = std::midpoint(low, high); // C++20
    if (a[mid] < key) {
      low = mid + 1;
    } else if (a[mid] > key) {
      high = mid - 1;
    } else {
      return true;
    }
  }
  return false;
}

int main(int argc, char* argv[])
{
  using namespace std;

  cout << "Running linear..." << endl;
  auto a = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto found_a = linear_search(a, 7);
  if (found_a) {
    cout << "Found element" << endl;
  } else {
    cout << "Didn't find element" << endl;
  }

  cout << "Running binary..." << endl;
  auto b = std::vector{1, 2, 3, 4, 5, 6, 7, 8};
  auto found_b = binary_search(a, 7);
  if (found_b) {
    cout << "Found element" << endl;
  } else {
    cout << "Didn't find element" << endl;
  }
  
  return EXIT_SUCCESS;
}

