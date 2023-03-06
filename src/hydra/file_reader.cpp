/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2022 Blender Foundation */

#include "hydra/file_reader.h"
#include "hydra/camera.h"
#include "hydra/render_delegate.h"

#include "util/path.h"
#include "util/unique_ptr.h"

#include "scene/background.h"
#include "scene/camera.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/scene.h"
#include "scene/shader_graph.h"
#include "scene/shader_nodes.h"

#include "app/cycles_xml.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hd/dirtyList.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/rprimCollection.h>
#include <pxr/imaging/hd/task.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usdImaging/usdImaging/delegate.h>

HDCYCLES_NAMESPACE_OPEN_SCOPE

/* Dummy task whose only purpose is to provide render tag tokens to the render index. */
class DummyHdTask : public HdTask {
 public:
  DummyHdTask(HdSceneDelegate *delegate, SdfPath const &id)
      : HdTask(id), tags({HdRenderTagTokens->geometry, HdRenderTagTokens->render})
  {
  }

 protected:
  void Sync(HdSceneDelegate *delegate, HdTaskContext *ctx, HdDirtyBits *dirtyBits) override
  {
  }

  void Prepare(HdTaskContext *ctx, HdRenderIndex *render_index) override
  {
  }

  void Execute(HdTaskContext *ctx) override
  {
  }

  const TfTokenVector &GetRenderTags() const override
  {
    return tags;
  }

  TfTokenVector tags;
};

Shader *createMeshShader(Scene *scene, UsdGeomMesh const &usdMesh)
{
  auto displayColorAttr = usdMesh.GetDisplayColorAttr();
  pxr::GfVec3f displayColor(0.8, 0.8, 0.8);
  if (displayColorAttr) {
    pxr::VtVec3fArray displayColors;
    // std::cout << displayColorAttr.GetTypeName() << std::endl;
    if (displayColorAttr.Get(&displayColors) && displayColors.size() > 0) {
      displayColor = displayColors[0];
      // std::cout << "Display color: " << displayColor[0] << ", " << displayColor[1] << ", "
      //           << displayColor[2] << std::endl;
      // std::exit(1);
    }
  }

  std::string xmlData = R"(<cycles>
    <shader name="shader)" +
                        std::to_string(scene->shaders.size()) + R"(">
      <glossy_bsdf name="floor_closure" distribution="beckmann" roughness="0.8" color=")" +
                        std::to_string(displayColor[0]) + " " + std::to_string(displayColor[1]) +
                        " " + std::to_string(displayColor[2]) + R"("/>
      <connect from="floor_closure bsdf" to="output surface" />
    </shader>
   </cycles>
  )";

  if (usdMesh.GetPath().GetString().find("ocean_geo") != std::string::npos) {
    displayColor = pxr::GfVec3f(0.6, 0.6, 0.8);
    xmlData = R"(<cycles>
      <shader name="shader)" +
              std::to_string(scene->shaders.size()) + R"(">
        <glass_bsdf name="floor_closure" distribution="beckmann" IOR="1.33" roughness="0.1" color=")" +
              std::to_string(displayColor[0]) + " " + std::to_string(displayColor[1]) + " " +
              std::to_string(displayColor[2]) + R"("/>
        <connect from="floor_closure bsdf" to="output surface" />
      </shader>
    </cycles>
    )";
  }

  xmlReadFromString(scene, xmlData);
  return scene->shaders.back();
}

void readMesh(ccl::Scene *scene,
              UsdGeomMesh const &usdMesh,
              glm::mat4 const &xform,
              Node *meshShader,
              std::unordered_map<std::string, Mesh *> &meshes)
{
  /* create mesh */

  auto const meshName = usdMesh.GetPrim().GetName().GetString();
  auto const addInstanceToScene = [&xform, &scene](Mesh *mesh) {
    Object *object = new Object();
    object->set_geometry(mesh);
    Transform tfm;
    const auto transpose = glm::transpose(xform);
    memcpy(&tfm, &transpose[0][0], sizeof(tfm));
    object->set_tfm(tfm);
    scene->objects.push_back(object);
  };

  if (meshes.find(meshName) != meshes.end()) {
    // Mesh alread exists, this is an instance
    addInstanceToScene(meshes.find(meshName)->second);
    return;
  }

  meshShader = createMeshShader(scene, usdMesh);

  Mesh *mesh = new Mesh();
  scene->geometry.push_back(mesh);
  meshes.insert({meshName, mesh});

  /* Create object. */
  addInstanceToScene(mesh);

  array<Node *> used_shaders = mesh->get_used_shaders();
  used_shaders.push_back_slow(meshShader);
  mesh->set_used_shaders(used_shaders);

  /* read state */
  int shader = 0;
  bool smooth = true;

  /* read vertices and polygons */
  pxr::VtArray<pxr::GfVec3f> P;
  pxr::VtArray<int> verts;
  pxr::VtArray<int> nverts;
  // vector<float> UV;
  // vector<int> verts, nverts;
  usdMesh.GetPointsAttr().Get(&P);
  usdMesh.GetFaceVertexIndicesAttr().Get(&verts);
  usdMesh.GetFaceVertexCountsAttr().Get(&nverts);

  array<float3> P_array{P.size()};
  for (size_t i = 0; i < P.size(); i++) {
    P_array[i].x = P[i].GetArray()[0];
    P_array[i].y = P[i].GetArray()[1];
    P_array[i].z = P[i].GetArray()[2];
  }

  /* create vertices */
  size_t num_triangles = 0;
  for (size_t i = 0; i < nverts.size(); i++)
    num_triangles += nverts[i] - 2;
  mesh->reserve_mesh(mesh->get_verts().size(), num_triangles);
  mesh->set_verts(P_array);

  /* create triangles */
  int index_offset = 0;

  for (size_t i = 0; i < nverts.size(); i++) {
    for (int j = 0; j < nverts[i] - 2; j++) {
      int v0 = verts[index_offset];
      int v1 = verts[index_offset + j + 1];
      int v2 = verts[index_offset + j + 2];

      // assert(v0 < (int)P.size());
      // assert(v1 < (int)P.size());
      // assert(v2 < (int)P.size());
      if (v0 >= P.size() || v1 >= P.size() || v2 >= P.size()) {
        continue;
      }

      mesh->add_triangle(v0, v1, v2, shader, smooth);
    }

    index_offset += nverts[i];
  }

  // if (xml_read_float_array(UV, node, "UV")) {
  //   ustring name = ustring("UVMap");
  //   Attribute *attr = mesh->attributes.add(ATTR_STD_UV, name);
  //   float2 *fdata = attr->data_float2();

  //   /* loop over the triangles */
  //   index_offset = 0;
  //   for (size_t i = 0; i < nverts.size(); i++) {
  //     for (int j = 0; j < nverts[i] - 2; j++) {
  //       int v0 = index_offset;
  //       int v1 = index_offset + j + 1;
  //       int v2 = index_offset + j + 2;

  //       assert(v0 * 2 + 1 < (int)UV.size());
  //       assert(v1 * 2 + 1 < (int)UV.size());
  //       assert(v2 * 2 + 1 < (int)UV.size());

  //       fdata[0] = make_float2(UV[v0 * 2], UV[v0 * 2 + 1]);
  //       fdata[1] = make_float2(UV[v1 * 2], UV[v1 * 2 + 1]);
  //       fdata[2] = make_float2(UV[v2 * 2], UV[v2 * 2 + 1]);
  //       fdata += 3;
  //     }

  //     index_offset += nverts[i];
  //   }
  // }
}

void updateXform(glm::mat4 &xform, UsdGeomXform const &xformPrim)
{
  pxr::GfMatrix4d pxrXform;
  bool resetXformStack = false;
  bool success = xformPrim.GetLocalTransformation(&pxrXform, &resetXformStack);
  if (!success) {
    std::cerr << "Error with GetLocalTransformation: " << xformPrim.GetPath().GetString()
              << std::endl;
    std::exit(1);
  }
  glm::mat4 M;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      M[i][j] = pxrXform.data()[4 * i + j];
    }
  }
  xform *= M;
}

void processPointInstancer(Scene *scene,
                           UsdStage const &stage,
                           UsdGeomPointInstancer const &instancer,
                           std::unordered_map<std::string, Mesh *> const &meshes)
{
  pxr::VtArray<pxr::GfVec3f> positions;
  pxr::SdfPathVector protoTypes;
  std::vector<Mesh *> protoMeshes;
  pxr::VtArray<int> protoIndices;
  instancer.GetPositionsAttr().Get(&positions);
  instancer.GetPrototypesRel().GetTargets(&protoTypes);
  instancer.GetProtoIndicesAttr().Get(&protoIndices);

  for (auto const &path : protoTypes) {
    auto const &protoPrim = stage.GetPrimAtPath(path);
    if (!protoPrim.IsValid()) {
      std::cerr << "Unabled to find prim at USD path: " << path.GetAsString() << std::endl;
      return;
    }
    // std::cout << "PointInstancer prototype: " << path.GetString() << " --> "
    //           << protoPrim.GetPrototype().GetPath().GetString() << std::endl;
    auto const entry = meshes.find(protoPrim.GetPrototype().GetName());
    if (entry == meshes.end()) {
      std::cerr << "Unabled to find Cycles mesh for USD prim at path: " << path.GetAsString()
                << std::endl;
      return;
    }
    protoMeshes.push_back(entry->second);
  }

  if (positions.size() != protoIndices.size()) {
    std::cerr << "Attributes have differing sizes" << std::endl;
    std::exit(1);
  }
  for (size_t i = 0; i < positions.size(); i++) {
    // TODO(jberchtold) use orientation and scale attribute
    auto const &p = positions[i];
    auto *mesh = protoMeshes[protoIndices[i]];

    auto const xform = glm::translate(glm::mat4(1), glm::vec3(p[0], p[1], p[2]));

    Object *object = new Object();
    object->set_geometry(mesh);
    Transform tfm;
    const auto transpose = glm::transpose(xform);
    memcpy(&tfm, &transpose[0][0], sizeof(tfm));
    object->set_tfm(tfm);
    scene->objects.push_back(object);
  }
}

void processPointInstancers(Scene *scene,
                            UsdStage const &stage,
                            std::vector<UsdGeomPointInstancer> const &pointInstancers,
                            std::unordered_map<std::string, Mesh *> const &meshes)
{
  // TODO(jberchtold) need to store xform when saving point instancers so they keep their original
  // transform
  for (auto &pointInstancer : pointInstancers) {
    processPointInstancer(scene, stage, pointInstancer, meshes);
  }
}

const std::string cameraName = "/island/cam/shotCam";

void printRecursive(UsdPrim const &prim)
{
  std::cout << prim.GetPath() << ": " << prim.GetTypeName() << std::endl;
  for (auto const &prim : prim.GetChildren()) {
    printRecursive(prim);
  }
}

static std::unordered_set<std::string> typesFound;

void traverseUsd(ccl::Scene *scene,
                 UsdStage const &stage,
                 UsdPrim const &prim,
                 glm::mat4 xform,
                 Node *meshShader,
                 std::unordered_map<std::string, Mesh *> &meshes,
                 std::vector<UsdGeomPointInstancer> &pointInstancers)
{
  if (UsdGeomImageable imageable = UsdGeomImageable(prim)) {
    auto visAttr = imageable.GetVisibilityAttr();
    if (visAttr) {
      pxr::TfToken visibility;
      visAttr.Get(&visibility);
      if (visibility == pxr::UsdGeomTokens->invisible) {
        return;  // Don't process invisible prims
      }
    }

    pxr::UsdAttribute purpose_attr = imageable.GetPurposeAttr();
    if (purpose_attr) {
      pxr::TfToken purpose;
      purpose_attr.Get(&purpose);
      if (purpose != pxr::UsdGeomTokens->render && purpose != pxr::UsdGeomTokens->default_) {
        return;  // Don't process prims that aren't for rendering
      }
    }
  }

  // if (prim.IsInstance()) {
  //   std::cout << "IsInstance: " << prim.GetPath().GetString() << std::endl;
  //   printRecursive(prim.GetPrototype());
  //   return;
  // }
  std::cout << prim.GetPath().GetAsString() << ": " << prim.GetTypeName().GetString() << std::endl;
  if (prim.IsA<UsdGeomCamera>()) {
    // if (prim.GetPath().GetAsString() == cameraName) {
    //   HdCyclesCamera camera(prim.GetPath());
    //   std::cout << "Camera: " << prim.GetPath().GetAsString().c_str() << std::endl;
    //   camera.ApplyCameraSettings(nullptr, scene->camera);
    //   scene->camera->need_flags_update = true;
    //   scene->camera->update(scene);
    // }
  }
  else if (UsdGeomMesh mesh = UsdGeomMesh(prim)) {
    readMesh(scene, mesh, xform, meshShader, meshes);
  }
  else if (UsdGeomXform xformPrim = UsdGeomXform(prim)) {
    updateXform(xform, xformPrim);
  }
  else if (UsdGeomPointInstancer instancerPrim = UsdGeomPointInstancer(prim)) {
    pointInstancers.push_back(instancerPrim);
  }
  else if (prim.GetTypeName().size() > 0) {
    // std::cerr << "Unsupported USD node type: " << prim.GetTypeName() << std::endl;
    // std::exit(1);
    typesFound.insert(prim.GetTypeName().GetString());
  }

  for (const auto &child : prim.GetFilteredChildren(UsdTraverseInstanceProxies())) {
    traverseUsd(scene, stage, child, xform, meshShader, meshes, pointInstancers);
  }
}

void initBaseScene(Scene *scene)
{
  xml_read_file(scene, "base_scene.xml");
}

void convertFromUSD(ccl::Scene *scene, UsdStageRefPtr stage)
{
  initBaseScene(scene);
  bool isMoana = true;
  const glm::mat4 moanaTransform = glm::scale(glm::mat4(0.001f), glm::vec3(1, 1, -1));
  const glm::mat4 otherTransform = glm::scale(
      glm::rotate(glm::mat4(0.01f), static_cast<float>(M_PI / 2.f), glm::vec3(1, 0, 0)),
      glm::vec3(1, 1, -1));
  glm::mat4 baseUsdTransform = isMoana ? moanaTransform : otherTransform;
  std::unordered_map<std::string, Mesh *> meshes;
  std::vector<UsdGeomPointInstancer> pointInstancers;

  Shader *meshShader = scene->default_surface;
  for (auto const shader : scene->shaders) {
    std::cout << "Shader: " << shader->name.c_str() << std::endl;
    if (shader->name == "floor") {
      meshShader = shader;
    }
  }

  auto const processPrim = [&](UsdPrim const &prim) {
    traverseUsd(scene, *stage, prim, baseUsdTransform, meshShader, meshes, pointInstancers);
  };
  processPrim(stage->GetPseudoRoot());

  processPointInstancers(scene, *stage, pointInstancers, meshes);

  for (auto const &s : typesFound) {
    std::cerr << "Unsupported USD Type: " << s << std::endl;
  }
  std::cout << "Meshes: " << meshes.size() << std::endl;
}

void HdCyclesFileReader::read(Session *session, const char *filepath, const bool use_camera)
{
  /* Initialize USD. */
  PlugRegistry::GetInstance().RegisterPlugins(path_get("usd"));

  /* Open Stage. */
  UsdStageRefPtr stage = UsdStage::Open(filepath);
  if (!stage) {
    fprintf(stderr, "%s read error\n", filepath);
    return;
  }

  convertFromUSD(session->scene, stage);

  //   initBaseScene(session->scene);
  //   /* Init paths. */
  //   SdfPath root_path = SdfPath::AbsoluteRootPath();
  //   SdfPath task_path("/_hdCycles/DummyHdTask");

  //   /* Create render delegate. */
  //   HdRenderSettingsMap settings_map;
  //   settings_map.insert(std::make_pair(HdCyclesRenderSettingsTokens->stageMetersPerUnit,
  //                                      VtValue(UsdGeomGetStageMetersPerUnit(stage))));

  //   HdCyclesDelegate render_delegate(settings_map, session, true);

  //   /* Create render index and scene delegate. */
  //   unique_ptr<HdRenderIndex> render_index(HdRenderIndex::New(&render_delegate, {}));
  //   std::cout << root_path << ", " << filepath << std::endl;
  //   unique_ptr<UsdImagingDelegate> scene_delegate = make_unique<UsdImagingDelegate>(
  //       render_index.get(), root_path);

  //   /* Add render tags and collection to render index. */
  //   HdRprimCollection collection(HdTokens->geometry, HdReprSelector(HdReprTokens->smoothHull));
  //   collection.SetRootPath(root_path);

  //   // render_index->InsertTask<DummyHdTask>(scene_delegate.get(), task_path);

  // #if PXR_VERSION < 2111
  //   HdDirtyListSharedPtr dirty_list = std::make_shared<HdDirtyList>(collection,
  //                                                                   *(render_index.get()));
  //   render_index->EnqueuePrimsToSync(dirty_list, collection);
  // #else
  //   render_index->EnqueueCollectionToSync(collection);
  // #endif

  //   /* Create prims. */
  //   const UsdPrim &stage_root = stage->GetPseudoRoot();
  //   // scene_delegate->Populate(stage_root.GetStage()->GetPrimAtPath(root_path), {});

  //   /* Sync prims. */
  //   HdTaskContext task_context;
  //   HdTaskSharedPtrVector tasks;
  //   tasks.push_back(render_index->GetTask(task_path));

  //   render_index->SyncAll(&tasks, &task_context);
  //   render_delegate.CommitResources(&render_index->GetChangeTracker());

  //   /* Use first camera in stage.
  //    * TODO: get camera from UsdRender if available. */
  //   if (use_camera) {
  //     for (UsdPrim const &prim : stage->Traverse()) {
  //       if (prim.IsA<UsdGeomCamera>()) {
  //         HdSprim *sprim = render_index->GetSprim(HdPrimTypeTokens->camera, prim.GetPath());
  //         if (sprim) {
  //           HdCyclesCamera *camera = dynamic_cast<HdCyclesCamera *>(sprim);
  //           camera->ApplyCameraSettings(render_delegate.GetRenderParam(),
  //           session->scene->camera); break;
  //         }
  //       }
  //     }
  //   }

  //   std::cout << "Objects: " << session->scene->objects.size() << std::endl;
}

HDCYCLES_NAMESPACE_CLOSE_SCOPE
