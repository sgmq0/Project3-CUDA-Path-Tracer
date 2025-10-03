#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::subdivide(int nodeIdx) {
    BVHNode& node = bvhNodes[nodeIdx];
    if (node.primCount <= 2) {
        node.isLeaf = true;
        return;
    }

    // find longest axis and split position
    glm::vec3 extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float splitPos = (node.aabbMin[axis] + node.aabbMax[axis]) * 0.5f;

    // split group in two halves
    int i = node.firstPrim;
    int j = i + node.primCount - 1;

    while (i <= j)
    {
        if (triangles[i].centroid[axis] < splitPos)
            i++;
        else 
            swap(triangles[i], triangles[j--]);
    }

    // create child nodes for each half
    int leftCount = i - node.firstPrim;
    if (leftCount == 0 || leftCount == node.primCount) {
		node.isLeaf = true;
        return;
    }

    // create child nodes
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;
    node.leftChild = leftChildIdx;
    bvhNodes[leftChildIdx].firstPrim = node.firstPrim;
    bvhNodes[leftChildIdx].primCount = leftCount;
    bvhNodes[rightChildIdx].firstPrim = i;
    bvhNodes[rightChildIdx].primCount = node.primCount - leftCount;
    node.primCount = 0;

	// update bounds of each child
    updateNodeBounds(leftChildIdx); 
    updateNodeBounds(rightChildIdx);

    // recurse and subdivide
    subdivide(leftChildIdx);
	subdivide(rightChildIdx);
}

void Scene::updateNodeBounds(int nodeIdx) {
	BVHNode& node = bvhNodes[nodeIdx];

    node.aabbMin = glm::vec3(INFINITY, INFINITY, INFINITY);
    node.aabbMax = glm::vec3(-INFINITY, -INFINITY, -INFINITY);

    for (int i = 0; i < node.primCount; i++) {
        Triangle leafTri = triangles[i + node.firstPrim];

        node.aabbMin = min(node.aabbMin, leafTri.v0);
        node.aabbMin = min(node.aabbMin, leafTri.v1);
        node.aabbMin = min(node.aabbMin, leafTri.v2);
        node.aabbMax = max(node.aabbMax, leafTri.v0);
        node.aabbMax = max(node.aabbMax, leafTri.v1);
        node.aabbMax = max(node.aabbMax, leafTri.v2);
    }
    
}

void Scene::buildBVH(int& nodeIdx) {
    BVHNode& root = bvhNodes[nodeIdx];
    root.leftChild = 0;
    root.rightChild = 0;
    root.firstPrim = 0;
    root.primCount = numTriangles;
    updateNodeBounds(nodeIdx);
    subdivide(nodeIdx);
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    numTriangles = 0;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};

        float roughness = p["ROUGHNESS"];  // basically specular degree
        float indexOfRefraction = p["IOR"]; // for glass material
		float transmission = p["TRANSMISSION"];   // for glass material
		float emittance = p["EMITTANCE"]; // light intensity
        const auto& albedo = p["ALBEDO"];   // base color

		newMaterial.roughness = roughness;
		newMaterial.indexOfRefraction = indexOfRefraction;
        newMaterial.transmission = transmission;
		newMaterial.emittance = emittance;
        newMaterial.color = glm::vec3(albedo[0], albedo[1], albedo[2]);

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "mesh")
        {
            std::string file = p["FILENAME"];

            //LoadGLTF(file, triangles, numTriangles);

            int start;
            int triCount;
            glm::vec3 bboxMin;
            glm::vec3 bboxMax;

		    LoadGLTF(file, triangles, numTriangles, start, triCount, bboxMin, bboxMax);

            newGeom.type = MESH;
            newGeom.startIdx = start;
            newGeom.numTriangles = triCount;
            newGeom.bboxMin = bboxMin;
            newGeom.bboxMax = bboxMax;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }

    // initialize bvh stuff
	std::cout << "Total number of triangles: " << numTriangles << "\n";

	// calculate centroids for all triangles
    for (int i = 0; i < numTriangles; i++) {
        triangles[i].centroid = (triangles[i].v0 + triangles[i].v1 + triangles[i].v2) / 3.0f;
	}

	bvhNodes = std::vector<BVHNode>(numTriangles * 2 - 1); 
    int rootNodeIdx = 0; 
    nodesUsed = 1;

    // build bvh once all tris are loaded
    buildBVH(rootNodeIdx);
	std::cout << "BVH build complete. Total nodes used: " << nodesUsed << "\n";

    //camera stuff (given in base code)
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
