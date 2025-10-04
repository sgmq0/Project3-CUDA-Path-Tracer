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

__host__ __device__ glm::vec3 reflect_direction(glm::vec3 normal, glm::vec3 direction) {
    return glm::reflect(glm::normalize(direction), normal);
}

__host__ __device__ glm::vec3 refract_direction(float ior, glm::vec3 normal, glm::vec3 direction) {
    float cosTheta = glm::dot(normal, direction);
    float eta = (cosTheta > 0) ? (ior / 1.0f) : (1.0f / ior);

    glm::vec3 N = (cosTheta > 0) ? -normal : normal;
    glm::vec3 refractDirection = glm::normalize(glm::refract(direction, N, eta));

    return refractDirection;
}

__host__ __device__ glm::vec3 fresnelDielectricEval(float cosThetaI) {
    // We will hard-code the indices of refraction to be
    // those of glass
    float etaI = 1.;
    float etaT = 1.55;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    bool entering = cosThetaI > 0.;
    if (!entering) {
        etaI = 1.55;
        etaT = 1.;
        cosThetaI = abs(cosThetaI);
    }

    float sinThetaI = sqrt(glm::max(0., 1. - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1) return glm::vec3(1.);

    float cosThetaT = sqrt(glm::max(0., 1. - sinThetaT * sinThetaT));

    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));

    return glm::vec3((Rparl * Rparl + Rperp * Rperp) / 2.0);
}


__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    glm::vec3 newDirection = glm::vec3();

    if (m.type == DIFFUSE) {
        // lambert 
        newDirection = normalize(calculateRandomDirectionInHemisphere(normal, rng));
    }
    else if (m.type == TRANSMISSIVE) {
		// mix of reflection and refraction
        thrust::uniform_real_distribution<float> u01(0, 1);
		float rand = u01(rng);

		// compute fresnel (from pbr textbook)
        float cosTheta = glm::dot(normal, pathSegment.ray.direction);
        float eta = (cosTheta > 0) ? (m.indexOfRefraction / 1.0f) : (1.0f / m.indexOfRefraction);

        cosTheta = glm::clamp(cosTheta, -1.f, 1.f);
        if (cosTheta < 0) {
            eta = 1.f / eta;
            cosTheta = -cosTheta;
        }

        float fresnel;
        float sin2Theta_i = 1.f - cosTheta * cosTheta;
        float sin2Theta_t = sin2Theta_i / (eta * eta);
        if (sin2Theta_t >= 1)
            fresnel = 1.f;
        else {
            float cosTheta_t = sqrt(1.f - sin2Theta_t);
            float r_parl = (eta * cosTheta - cosTheta_t) / (eta * cosTheta + cosTheta_t);
            float r_perp = (cosTheta - eta * cosTheta_t) / (cosTheta + eta * cosTheta_t);
            fresnel = (r_parl * r_parl + r_perp * r_perp) / 2.f;
        }

        if (rand < fresnel) {
            glm::vec3 refl = reflect_direction(normal, pathSegment.ray.direction);
			newDirection = refl;
        }
        else {
            glm::vec3 refr = refract_direction(m.indexOfRefraction, normal, pathSegment.ray.direction);
			newDirection = refr;
		}
    } else if (m.type == SPECULAR) {
		// pure refraction
        newDirection = reflect_direction(normal, pathSegment.ray.direction);
    }

    // set new ray
    pathSegment.color *= m.color;
    pathSegment.ray.origin = intersect + EPSILON * newDirection;
    pathSegment.ray.direction = newDirection;

    // reduce number of bounces
    if (pathSegment.remainingBounces >= 1)
      pathSegment.remainingBounces--;
}
