static bool eigen_did_assert = false;
#define eigen_assert(X)                                                                \
  if (!eigen_did_assert && !(X)) {                                                     \
    std::cout << "### Assertion raised in " << __FILE__ << ":" << __LINE__ << ":\n" #X \
              << "\n### The following would happen without assertions:\n";             \
    eigen_did_assert = true;                                                           \
  }

#include <iostream>
#include <cassert>
#include <Eigen/Eigen>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

using namespace Eigen;
using namespace std;

int main(int, char**) {
  cout.precision(3);
// intentionally remove indentation of snippet
{
struct pad {
  Index size() const { return out_size; }
  Index operator[](Index i) const { return std::max<Index>(0, i - (out_size - in_size)); }
  Index in_size, out_size;
};

Matrix3i A;
A.reshaped() = VectorXi::LinSpaced(9, 1, 9);
cout << "Initial matrix A:\n" << A << "\n\n";
MatrixXi B(5, 5);
B = A(pad{3, 5}, pad{3, 5});
cout << "A(pad{3,N}, pad{3,N}):\n" << B << "\n\n";

}
  return 0;
}
