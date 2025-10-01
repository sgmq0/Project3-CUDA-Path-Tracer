#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    /*if (!outside)
    {
        normal = -normal;
    }*/

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__
bool intersectRayTriangleMT(
    const glm::vec3& orig, const glm::vec3& dir,
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    float& t, float& u, float& v)
{
    glm::vec3 v0v1 = v1 - v0;
    glm::vec3 v0v2 = v2 - v0;
    glm::vec3 pvec = glm::cross(dir, v0v2);
    float det = glm::dot(v0v1, pvec);

    if (det < EPSILON) return false;
    if (fabs(det) < EPSILON) return false;

    float invDet = 1.0f / det;
    
    glm::vec3 tvec = orig - v0;
    u = invDet * glm::dot(tvec, pvec);
    if (u < 0.0f || u > 1.0f) return false;

    glm::vec3 qvec = glm::cross(tvec, v0v1);
    v = invDet * glm::dot(dir, qvec);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = invDet * glm::dot(v0v2, qvec);

    return t > EPSILON;
}

__host__ __device__ float meshIntersectionTest(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside, 
    Triangle* triangles,
    int numTriangles)
{
    float t = INFINITY;
    Triangle hitTri = triangles[0];

    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    for (int i = mesh.startIdx; i < mesh.endIdx; ++i) {
        Triangle tri = triangles[i];
        float tTemp, u, v;
        bool hit = intersectRayTriangleMT(ro, rd, tri.v0, tri.v1, tri.v2, tTemp, u, v);

        if (hit && tTemp < t) {
            t = tTemp;
            hitTri = tri;
        }
    }

    if (t < INFINITY) {
        // Calculate flat normal in object space
        glm::vec3 edge1 = hitTri.v1 - hitTri.v0;
        glm::vec3 edge2 = hitTri.v2 - hitTri.v0;
        glm::vec3 objectNormal = glm::normalize(glm::cross(edge1, edge2));

        // calculate intersection 
        intersectionPoint = ro + t * rd;

        glm::vec3 worldNormal = glm::normalize(glm::vec3(
            glm::transpose(glm::inverse(mesh.transform)) * glm::vec4(objectNormal, 0.0f)));

        normal = worldNormal;

        outside = glm::dot(rd, objectNormal) < 0.0f;
        if (!outside) {
            worldNormal = -worldNormal;
        }

        // world space distance
        glm::vec3 worldIntersection = multiplyMV(mesh.transform, glm::vec4(intersectionPoint, 1.0f));
        return glm::length(worldIntersection - r.origin);
    }

    return -1;
}