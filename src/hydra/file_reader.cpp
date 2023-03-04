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

#include <glm/glm.hpp>
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

void readMesh(ccl::Scene *scene,
              UsdGeomMesh const &usdMesh,
              glm::mat4 const &xform,
              Node *meshShader)
{
  /* create mesh */
  Mesh *mesh = new Mesh();
  scene->geometry.push_back(mesh);

  /* Create object. */
  Object *object = new Object();
  object->set_geometry(mesh);
  Transform tfm;
  const auto transpose = glm::transpose(xform);
  memcpy(&tfm, &transpose[0][0], sizeof(tfm));
  object->set_tfm(tfm);
  scene->objects.push_back(object);

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
  mesh->set_verts(P_array);

  size_t num_triangles = 0;
  for (size_t i = 0; i < nverts.size(); i++)
    num_triangles += nverts[i] - 2;
  mesh->reserve_mesh(mesh->get_verts().size(), num_triangles);

  /* create triangles */
  int index_offset = 0;

  for (size_t i = 0; i < nverts.size(); i++) {
    for (int j = 0; j < nverts[i] - 2; j++) {
      int v0 = verts[index_offset];
      int v1 = verts[index_offset + j + 1];
      int v2 = verts[index_offset + j + 2];

      assert(v0 < (int)P.size());
      assert(v1 < (int)P.size());
      assert(v2 < (int)P.size());

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
  xformPrim.GetLocalTransformation(&pxrXform, &resetXformStack);
  glm::mat4 M;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      M[i][j] = pxrXform.data()[4 * i + j];
    }
  }
  xform *= M;
}

void traverseUsd(ccl::Scene *scene, UsdPrim const &prim, glm::mat4 xform, Node *meshShader)
{
  std::cout << prim.GetPath().GetAsString() << ": " << prim.GetTypeName().GetString() << std::endl;
  if (prim.IsA<UsdGeomCamera>()) {
    HdCyclesCamera camera(prim.GetPath());
    std::cout << "Camera: " << prim.GetPath().GetAsString().c_str() << std::endl;
    camera.ApplyCameraSettings(nullptr, scene->camera);
    scene->camera->need_flags_update = true;
    scene->camera->update(scene);
  }
  else if (UsdGeomMesh mesh = UsdGeomMesh(prim)) {
    readMesh(scene, mesh, xform, meshShader);
  }
  else if (UsdGeomXform xformPrim = UsdGeomXform(prim)) {
    updateXform(xform, xformPrim);
  }
  else if (prim.GetTypeName().size() > 0) {
    std::cerr << "Unsupported USD node type: " << prim.GetTypeName() << std::endl;
    std::exit(1);
  }

  for (const auto &child : prim.GetChildren()) {
    traverseUsd(scene, child, xform, meshShader);
  }
}

Shader *createMeshShader(ccl::Scene *scene)
{
  Shader *shader = new Shader();

  ShaderGraph *graph = new ShaderGraph();

  // auto *diffuseBsdf = graph->create_node<DiffuseBsdfNode>();
  // diffuseBsdf->set_roughness(0.5f);
  // graph->add(diffuseBsdf);

  auto *diffuseBsdf = graph->create_node<EmissionNode>();
  diffuseBsdf->set_color({1.f, 1.f, 0.f});
  graph->add(diffuseBsdf);

  auto *outputNode = graph->create_node<OutputNode>();
  graph->add(outputNode);

  graph->connect(diffuseBsdf->outputs[0], outputNode->inputs[0] /*surface output*/);

  shader->set_graph(graph);
  shader->tag_update(scene);

  scene->shaders.push_back(shader);

  return shader;
}

Shader *createBackgroundShader(ccl::Scene *scene)
{
  shaderFromXML(R"(
    <background>
      <background name="bg" strength="2.0" color="0.2, 0.2, 0.2" />
      <connect from="bg background" to="output surface" />
    </background>
  )");
  Shader *shader = new Shader();

  ShaderGraph *graph = new ShaderGraph();

  auto *skyTextureNode = graph->create_node<SkyTextureNode>();
  skyTextureNode->set_sky_type(NodeSkyType::NODE_SKY_HOSEK);
  graph->add(skyTextureNode);

  auto *backgroundNode = graph->create_node<BackgroundNode>();
  backgroundNode->set_strength(8.0f);
  graph->add(backgroundNode);

  auto *outputNode = graph->create_node<OutputNode>();
  graph->add(outputNode);

  graph->connect(skyTextureNode->outputs[0], backgroundNode->inputs[0]);
  graph->connect(backgroundNode->outputs[0], outputNode->inputs[0]);

  shader->set_graph(graph);
  shader->tag_update(scene);

  scene->shaders.push_back(shader);

  return shader;
}

void createDefaultCamera(Scene *scene)
{
  Camera *cam = scene->camera;
  cam->set_full_width(800);
  cam->set_full_width(500);

  glm::mat4 m(1.0f);
  const auto transpose = glm::transpose(m);
  Transform tfm;
  memcpy(&tfm, &transpose[0][0], sizeof(tfm));

  cam->set_matrix(tfm);
  cam->need_flags_update = true;
  cam->update(scene);
}

void convertFromUSD(ccl::Scene *scene, UsdStageRefPtr stage)
{
  createDefaultCamera(scene);  // Overriden by cameras in the USD file, if any are present
  auto *meshShader = createMeshShader(scene);
  scene->default_background = createBackgroundShader(scene);
  scene->background->set_use_shader(true);
  scene->background->set_shader(scene->default_background);
  traverseUsd(scene, stage->GetPseudoRoot(), glm::mat4(1.0f), scene->default_surface);
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
  //   std::exit(0);

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
}

HDCYCLES_NAMESPACE_CLOSE_SCOPE
