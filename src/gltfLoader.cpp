// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.

#include <iostream>
#include "gltfLoader.h"

using namespace tinygltf;

bool LoadGLTF(const std::string& filename, 
    std::vector<Triangle>& triangles, 
    glm::mat4 transform, 
    int materialID
) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    const std::string ext = GetFilePathExtension(filename);

    bool ret = false;
    if (ext.compare("glb") == 0) {
        // assume binary glTF.
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename.c_str());
    }
    else {
        // assume ascii glTF.
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());
    }

    if (!warn.empty()) {
    std::cout << "glTF parse warning: " << warn << std::endl;
    }

    if (!err.empty()) {
    std::cerr << "glTF parse error: " << err << std::endl;
    }
    if (!ret) {
    std::cerr << "Failed to load glTF: " << filename << std::endl;
    return false;
    }

    std::cout << "loaded glTF file has:\n"
    << model.accessors.size() << " accessors\n"
    << model.animations.size() << " animations\n"
    << model.buffers.size() << " buffers\n"
    << model.bufferViews.size() << " bufferViews\n"
    << model.materials.size() << " materials\n"
    << model.meshes.size() << " meshes\n"
    << model.nodes.size() << " nodes\n"
    << model.textures.size() << " textures\n"
    << model.images.size() << " images\n"
    << model.skins.size() << " skins\n"
    << model.samplers.size() << " samplers\n"
    << model.cameras.size() << " cameras\n"
    << model.scenes.size() << " scenes\n"
    << model.lights.size() << " lights\n";

    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(transform)));

    for (const auto& gltfMesh : model.meshes) {
        std::cout << "Current mesh has " << gltfMesh.primitives.size() << " primitives:\n";

        // Create a mesh object
        Mesh loadedMesh;

        loadedMesh.name = gltfMesh.name;
        std::cout << "Current mesh name: " << gltfMesh.name << "\n";

        for (const auto& meshPrimitive : gltfMesh.primitives) {
            // vectors for vertex data
            std::vector<glm::vec3> positions;
            std::vector<glm::vec3> normals;
            std::vector<glm::vec2> UVs;

            // load positions
            const auto& positionAccessor = model.accessors[meshPrimitive.attributes.at("POSITION")];
            const auto& posBufferView = model.bufferViews[positionAccessor.bufferView];
            const auto& posBuffer = model.buffers[posBufferView.buffer];

            // calculate byte offset
            const unsigned char* posDataPtr = posBuffer.data.data() + posBufferView.byteOffset + positionAccessor.byteOffset;
            int vertexCount = positionAccessor.count;

            // parse positions
            for (int i = 0; i < vertexCount; ++i) {
                const float* pos = reinterpret_cast<const float*>(posDataPtr + i * sizeof(float) * 3);

                // transform position
                glm::vec4 posTrans = transform * glm::vec4(pos[0], pos[1], pos[2], 1.0);

                positions.emplace_back(posTrans[0], posTrans[1], posTrans[2]);
            }

            // load normals
            const auto& normalAccessor = model.accessors[meshPrimitive.attributes.at("NORMAL")];
            const auto& norBufferView = model.bufferViews[normalAccessor.bufferView];
            const auto& norBuffer = model.buffers[norBufferView.buffer];

            // calculate byte offset
            const unsigned char* norDataPtr = norBuffer.data.data() + norBufferView.byteOffset + normalAccessor.byteOffset;
            int normalCount = normalAccessor.count;

            // parse normals
            for (int i = 0; i < normalCount; ++i) {
                const float* nor = reinterpret_cast<const float*>(norDataPtr + i * sizeof(float) * 3);

                glm::vec3 norTrans = glm::normalize(normalMatrix * glm::vec3(nor[0], nor[1], nor[2]));

                normals.emplace_back(norTrans);
            }

            if (meshPrimitive.attributes.find("TEXCOORD_0") != meshPrimitive.attributes.end()) {
                // load UVs
                const auto& uvAccessor = model.accessors[meshPrimitive.attributes.at("TEXCOORD_0")];
                const auto& uvBufferView = model.bufferViews[uvAccessor.bufferView];
                const auto& uvBuffer = model.buffers[uvBufferView.buffer];

                // calculate byte offset
                const unsigned char* uvDataPtr = uvBuffer.data.data() + uvBufferView.byteOffset + uvAccessor.byteOffset;
                int uvCount = uvAccessor.count;

                // parse UVs
                for (int i = 0; i < uvCount; ++i) {
                    const float* uv = reinterpret_cast<const float*>(uvDataPtr + i * sizeof(float) * 2);

                    glm::vec2 texCoord = glm::vec2(uv[0], uv[1]);

                    UVs.emplace_back(texCoord);
                }
            }
            else {
                for (int i = 0; i < vertexCount; ++i) {
                    glm::vec2 texCoord = glm::vec2(0.f, 0.f);
                    UVs.emplace_back(texCoord);
                }
            }

            // load indices
            const auto& indexAccessor = model.accessors[meshPrimitive.indices];
            const auto& idxBufferView = model.bufferViews[indexAccessor.bufferView];
            const auto& idxBuffer = model.buffers[idxBufferView.buffer];

            const unsigned char* idxDataPtr = idxBuffer.data.data() + idxBufferView.byteOffset + indexAccessor.byteOffset;
            size_t indexCount = indexAccessor.count;

            std::vector<uint32_t> indices;

            std::cout << "Loaded " << (positions.size() / 3) << " vertices.\n";
            switch (indexAccessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                    const uint8_t* buf = reinterpret_cast<const uint8_t*>(idxDataPtr);
                    for (int i = 0; i < indexCount; ++i) indices.push_back(buf[i]);
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    const uint16_t* buf = reinterpret_cast<const uint16_t*>(idxDataPtr);
                    for (int i = 0; i < indexCount; ++i) indices.push_back(buf[i]);
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    const uint32_t* buf = reinterpret_cast<const uint32_t*>(idxDataPtr);
                    for (int i = 0; i < indexCount; ++i) indices.push_back(buf[i]);
                    break;
                }
                default:
                    std::cerr << "Unsupported index type: " << indexAccessor.componentType << "\n";
                    continue;
            }

            // group positions into triangles
            if (indices.size() % 3 != 0) {
                continue;
            }

            for (int i = 0; i < indices.size(); i += 3) {
                int i0 = indices[i];
                int i1 = indices[i + 1];
                int i2 = indices[i + 2];

                if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) {
                    continue;
                }

                // find centroid
                glm::vec3 centroid = (positions[i0] + positions[i1] + positions[i2]) / 3.0f;

                Triangle tri;
                Vertex v0 = { positions[i0], normals[i0], UVs[i0] };
                Vertex v1 = { positions[i1], normals[i1], UVs[i1] };
                Vertex v2 = { positions[i2], normals[i2], UVs[i2] };

                // push triangle back
                triangles.push_back({v0, v1, v2, centroid, materialID});
            }
        }
    }

    // iterate through all the textures
    

    std::cout << "Loaded " << triangles.size() << " triangles.\n";

    return true;

}
