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
Matrix4d X = Matrix4d::Random(4, 4);
Matrix4d A = X + X.transpose();
cout << "Here is a random symmetric 4x4 matrix:" << endl << A << endl;
Tridiagonalization<Matrix4d> triOfA(A);
Matrix4d pm = triOfA.packedMatrix();
cout << "The packed matrix M is:" << endl << pm << endl;
cout << "The diagonal and subdiagonal corresponds to the matrix T, which is:" << endl << triOfA.matrixT() << endl;

}
  return 0;
}
