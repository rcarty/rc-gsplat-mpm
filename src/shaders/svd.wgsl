// 3x3 SVD via Jacobi on F^T F

const I3 = mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

fn diag(s: vec3f) -> mat3x3f {
  return mat3x3f(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z);
}

// one Jacobi rotation zeroing A[p][q], V collects the eigenvectors
fn jacobi(A: ptr<function, mat3x3f>, V: ptr<function, mat3x3f>, p: i32, q: i32) {
  let apq = (*A)[q][p];
  if (abs(apq) < 1e-12) {
    return;
  }
  let theta = ((*A)[q][q] - (*A)[p][p]) / (2.0 * apq);
  var t = 1.0 / (abs(theta) + sqrt(theta * theta + 1.0));
  if (theta < 0.0) {
    t = -t;
  }
  let c = 1.0 / sqrt(t * t + 1.0);
  let s = t * c;
  var J = I3;
  J[p][p] = c;
  J[q][q] = c;
  J[q][p] = s;
  J[p][q] = -s;
  *A = transpose(J) * (*A) * J;
  *V = (*V) * J;
}

struct Svd {
  U: mat3x3f,
  s: vec3f,
  V: mat3x3f,
}

// U and V stay rotations, s.z takes the sign if F is inverted
fn svd3(F: mat3x3f) -> Svd {
  var A = transpose(F) * F;
  var V = I3;
  for (var k = 0; k < 5; k++) {
    jacobi(&A, &V, 0, 1);
    jacobi(&A, &V, 0, 2);
    jacobi(&A, &V, 1, 2);
  }
  var e = vec3f(A[0][0], A[1][1], A[2][2]);
  if (e.x < e.y) {
    e = e.yxz;
    let c = V[0];
    V[0] = V[1];
    V[1] = c;
  }
  if (e.x < e.z) {
    e = e.zyx;
    let c = V[0];
    V[0] = V[2];
    V[2] = c;
  }
  if (e.y < e.z) {
    e = e.xzy;
    let c = V[1];
    V[1] = V[2];
    V[2] = c;
  }
  if (determinant(V) < 0.0) {
    V[2] = -V[2];
  }
  var u0 = F * V[0];
  let l0 = length(u0);
  if (l0 < 1e-6) {
    u0 = vec3f(1.0, 0.0, 0.0);
  } else {
    u0 = u0 / l0;
  }
  var u1 = F * V[1];
  u1 = u1 - dot(u1, u0) * u0;
  let l1 = length(u1);
  if (l1 < 1e-6) {
    var a = vec3f(1.0, 0.0, 0.0);
    if (abs(u0.x) > 0.9) {
      a = vec3f(0.0, 1.0, 0.0);
    }
    u1 = normalize(cross(u0, a));
  } else {
    u1 = u1 / l1;
  }
  let u2 = cross(u0, u1);
  let U = mat3x3f(u0, u1, u2);
  let s = vec3f(dot(u0, F * V[0]), dot(u1, F * V[1]), dot(u2, F * V[2]));
  return Svd(U, s, V);
}
