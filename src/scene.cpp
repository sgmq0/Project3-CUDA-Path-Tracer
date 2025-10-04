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

void Scene::subdivide(BVHNode& node) {

    // leaf node
    if (node.primCount <= 4) {
        node.isLeaf = true;
        return;
    }

    // find min and max centroid
	glm::vec3 minCentroid = glm::vec3(INFINITY, INFINITY, INFINITY);
	glm::vec3 maxCentroid = glm::vec3(-INFINITY, -INFINITY, -INFINITY);
    for (int i = 0; i < node.primCount; i++) {
        Triangle tri = triangles[i + node.firstPrim];
        minCentroid = min(minCentroid, tri.centroid);
        maxCentroid = max(maxCentroid, tri.centroid);
	}
    glm::vec3 extent = maxCentroid - minCentroid;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;

    float splitPos = (minCentroid[axis] + maxCentroid[axis]) * 0.5f;

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
    node.rightChild = rightChildIdx;

	BVHNode leftChild;
    leftChild.leftChild = 0;
	leftChild.rightChild = 0;
    leftChild.firstPrim = node.firstPrim;
	leftChild.primCount = leftCount;
	leftChild.isLeaf = false;

	BVHNode rightChild;
	rightChild.leftChild = 0;
	rightChild.rightChild = 0;
	rightChild.firstPrim = i;
	rightChild.primCount = node.primCount - leftCount;
	rightChild.isLeaf = false;

	bvhNodes.push_back(leftChild);
	bvhNodes.push_back(rightChild);

	// update bounds of each child
    updateNodeBounds(bvhNodes[leftChildIdx]); 
    updateNodeBounds(bvhNodes[rightChildIdx]);

    // recurse and subdivide
    subdivide(bvhNodes[leftChildIdx]);
	subdivide(bvhNodes[rightChildIdx]);
}

void Scene::updateNodeBounds(BVHNode& node) {
    node.aabbMin = glm::vec3(INFINITY, INFINITY, INFINITY);
    node.aabbMax = glm::vec3(-INFINITY, -INFINITY, -INFINITY);

    for (int i = 0; i < node.primCount; i++) {
        Triangle leafTri = triangles[i + node.firstPrim];

        glm::vec3 v0 = positions[leafTri.v0];
        glm::vec3 v1 = positions[leafTri.v1];
        glm::vec3 v2 = positions[leafTri.v2];

        node.aabbMin = min(node.aabbMin, v0);
        node.aabbMin = min(node.aabbMin, v1);
        node.aabbMin = min(node.aabbMin, v2);
        node.aabbMax = max(node.aabbMax, v0);
        node.aabbMax = max(node.aabbMax, v1);
        node.aabbMax = max(node.aabbMax, v2);
    }
    
}

void Scene::buildBVH() {
    bvhNodes.push_back(BVHNode{});

    int nodeIdx = (int)bvhNodes.size() - 1;

	bvhNodes[nodeIdx].firstPrim = 0;
	bvhNodes[nodeIdx].primCount = numTriangles;
	bvhNodes[nodeIdx].isLeaf = false;

    updateNodeBounds(bvhNodes[nodeIdx]);
    subdivide(bvhNodes[nodeIdx]);
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

        const auto& albedo = p["ALBEDO"];   // base color
        newMaterial.color = glm::vec3(albedo[0], albedo[1], albedo[2]);

        const auto& type = p["TYPE"];
        if (type == "LIGHT") {
            newMaterial.type = LIGHT;

            float emittance = p["EMITTANCE"]; // light intensity
            newMaterial.emittance = emittance;
        } 
        else if (type == "DIFFUSE")
        {
            newMaterial.type = DIFFUSE;
        }
        else if (type == "SPECULAR")
        {
            newMaterial.type = SPECULAR;
        }
        else
        {
            newMaterial.type = TRANSMISSIVE;

            float indexOfRefraction = p["IOR"]; // for glass material
            newMaterial.indexOfRefraction = indexOfRefraction;
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        Geom newGeom;

        // set material ID
        newGeom.materialid = MatNameToID[p["MATERIAL"]];

        // find transform vec3s
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);

        // build transformation matrix
        newGeom.transform = utilityCore::buildTransformationMatrix(
        newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        const auto& type = p["TYPE"];
        if (type == "cube")
        {
            newGeom.type = CUBE;
            geoms.push_back(newGeom);
        }
        else if (type == "mesh")
        {
            std::string file = p["FILENAME"];

            // apply transformation matrix and material to loaded triangle
		    LoadGLTF(file, triangles, positions, newGeom.transform, newGeom.materialid);
            numTriangles = triangles.size();
        }
        else
        {
            newGeom.type = SPHERE;
            geoms.push_back(newGeom);
        }

    }

    // initialize bvh stuff
	std::cout << "Total number of triangles: " << numTriangles << "\n";

    bvhNodes = std::vector<BVHNode>();
    nodesUsed = 1;

    // build bvh once all tris are loaded
    buildBVH();
	std::cout << "BVH build complete. Total nodes used: " << nodesUsed << "\n";

    // print all centroids
    //for (int i = 0; i < numTriangles; i++) {
    //    std::cout << "Triangle " << i << " centroid: " << glm::to_string(triangles[i].centroid) << "\n";
    //}

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
