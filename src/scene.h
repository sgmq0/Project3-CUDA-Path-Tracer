#pragma once

#include "sceneStructs.h"
#include "gltfLoader.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void buildBVH(int& nodeIdx);
    void updateNodeBounds(int nodeIdx);
    void subdivide(int nodeIdx);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    // bvh stuff
    std::vector<Triangle> triangles;
    int numTriangles;
    std::vector<BVHNode> bvhNodes;
    int nodesUsed;
};
