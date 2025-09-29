#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float AbsCosTheta(glm::vec3 w) { return abs(w.z); }
__host__ __device__ float CosTheta(glm::vec3 w) { return w.z; }

__host__ __device__ void coordinateSystem(glm::vec3 v1, glm::vec3& v2, glm::vec3& v3) {
  if (abs(v1.x) > abs(v1.y))
    v2 = glm::vec3(-v1.z, 0.f, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
  else
    v2 = glm::vec3 (0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
  v3 = cross(v1, v2);
}

__host__ __device__ glm::mat3 localToWorld(glm::vec3 nor) {
  glm::vec3 tan, bit;
  coordinateSystem(nor, tan, bit);
  return glm::mat3(tan, bit, nor);
}

__host__ __device__ glm::mat3 worldToLocal(glm::vec3 nor) {
  return transpose(localToWorld(nor));
}

__host__ __device__ glm::vec3 faceforward(glm::vec3 n, glm::vec3 v) {
  return (dot(n, v) < 0.f) ? -n : n;
}

__host__ __device__ bool Refract(glm::vec3 wi, glm::vec3 n, float eta, glm::vec3& wt) {
  // Compute cos theta using Snell's law
  float cosThetaI = dot(n, wi);
  float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
  float sin2ThetaT = eta * eta * sin2ThetaI;

  // Handle total internal reflection for transmission
  if (sin2ThetaT >= 1) return false;
  float cosThetaT = sqrt(1 - sin2ThetaT);
  wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
  return true;
}

__host__ __device__ glm::vec3 sample_f_specular_trans(glm::vec3 albedo, glm::vec3 nor, glm::vec3 wo,
                glm::vec3& wiW, int& sampledType) {

  float etaA = 1.f;
  float etaB = 1.55f;

  //took this all from pbr book
  bool entering = CosTheta(wo) > 0;
  float etaI = entering ? etaA : etaB;
  float etaT = entering ? etaB : etaA;

  glm::vec3 wi;

  //check for total internal reflection
  if (!Refract(wo, faceforward(glm::vec3(0., 0., 1.), wo), etaI / etaT, wi)) {
    return glm::vec3(0.);
  }

  sampledType = 1;

  wiW = localToWorld(nor) * wi;
  return albedo / AbsCosTheta(wi);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{

    glm::vec3 newOrigin = glm::vec3();
    glm::vec3 newDirection = glm::vec3();

    // lambert
    if (m.hasRefractive == 0 && m.hasReflective == 0) {
        pathSegment.color *= m.color;
        newOrigin = intersect + EPSILON * normal;
        newDirection = normalize(calculateRandomDirectionInHemisphere(normal, rng));
    }
    else if (m.hasReflective) {
      // compute reflected ray
      glm::vec3 reflectedRay = glm::reflect(glm::normalize(pathSegment.ray.direction), normal);

      pathSegment.color *= m.color;
      newOrigin = intersect + EPSILON * normal;
      newDirection = reflectedRay;
    }
    else if (m.hasRefractive) {
      float cosTheta = glm::dot(normal, pathSegment.ray.direction);
      float ior = 1.55f;
      float eta = (cosTheta > 0) ? (ior / 1.0f) : (1.0f / ior);

      glm::vec3 N = (cosTheta > 0) ? -normal : normal;
      glm::vec3 refractDirection = glm::normalize(glm::refract(pathSegment.ray.direction, N, eta));

      newDirection = refractDirection;
      newOrigin = intersect + EPSILON * newDirection;
      pathSegment.color *= m.color;
    }

    // set new ray
    pathSegment.ray.origin = newOrigin;
    pathSegment.ray.direction = newDirection;

    // reduce number of bounces
    if (pathSegment.remainingBounces >= 1)
      pathSegment.remainingBounces--;
}
