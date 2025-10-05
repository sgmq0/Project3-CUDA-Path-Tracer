#pragma once

#include "sceneStructs.h"
#include "gltfLoader.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void buildBVH();
    void updateNodeBounds(BVHNode& node);
    void subdivide(BVHNode& node);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    // bvh stuff
    std::vector<Triangle> triangles;
    std::vector<BVHNode> bvhNodes;
    std::vector<MyTexture> textures;
    std::vector<cudaTextureObject_t> cudaTextures;
    int numTriangles;
    int nodesUsed;
};
