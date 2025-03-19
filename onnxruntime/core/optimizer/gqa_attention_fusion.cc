// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gqa_attention_fusion.h"
#include <cmath>
#include "core/common/logging/logging.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

struct GQAParameters {
  int64_t batch_size_;
  int64_t num_heads_;
  int64_t kv_num_heads_;
  int64_t head_size_;
  int64_t seq_length_;
  int64_t past_seq_length_;
  float scale_;
};

namespace {
bool ValidateReshapeShape(
    Graph& graph,
    const NodeArg& reshape_shape,
    const std::initializer_list<int64_t>& expected_shape_values) {
  InlinedVector<int64_t> reshape_shape_temp;
  if (!optimizer_utils::AppendTensorFromInitializer(graph, reshape_shape,
                                                    reshape_shape_temp) ||
      reshape_shape_temp.size() != expected_shape_values.size()) {
    return false;
  }
  int index = 0;
  for (auto& expected_shape_value : expected_shape_values) {
    if (expected_shape_value > 0 &&
        reshape_shape_temp[index] != expected_shape_value) {
      return false;
    }
    ++index;
  }
  return true;
}

void MatchKVExpand(const Node* start_node, std::vector<std::reference_wrapper<const Node>>& node_lists, const logging::Logger& logger) {
  if (start_node->OpType().compare("ScatterND") == 0) {
    node_lists.push_back(*start_node);
  } else if (start_node->OpType().compare("Reshape") == 0) {
    std::vector<graph_utils::EdgeEndToMatch> present_kv_expand_path{
        {0, 0, "Expand", {8, 13}, kOnnxDomain},
        {0, 0, "Reshape", {5, 13, 21}, kOnnxDomain},
        {0, 0, "ScatterND", {13, 16, 18}, kOnnxDomain},
    };
    std::vector<const Node::EdgeEnd*> result;
    if (!graph_utils::FindPath(*start_node, true, present_kv_expand_path, result,
                               logger)) {
      return;
    }
    node_lists.push_back(*start_node);
    node_lists.push_back(result[0]->GetNode());
    node_lists.push_back(result[1]->GetNode());
    node_lists.push_back(result[2]->GetNode());
  }
}

bool CheckNodesInOutputPath(Graph& graph,
                            const Node& reshape,
                            const Node& transpose,
                            GQAParameters& gqa_params) {
  if (!optimizer_utils::CheckOutputEdges(graph, transpose, 1)) {
    LOGS_DEFAULT(WARNING) << "Output edge count not expected for Transpose in "
                             "output path, expected 1, got "
                          << transpose.GetOutputEdgesCount();
    return false;
  }

  InlinedVector<int64_t> perm;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(transpose, "perm", perm) &&
        perm.size() == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1 &&
        perm[3] == 3)) {
    return false;
  }

  const NodeArg& output_reshaped = *(reshape.OutputDefs()[0]);
  if (!optimizer_utils::IsShapeKnownOnAllDims(output_reshaped, 3)) {
    LOGS_DEFAULT(WARNING) << "Unknown GQA output_reshaped shape";
    return false;
  }
  if (output_reshaped.Shape()->dim(0).dim_value() != gqa_params.batch_size_ ||
      output_reshaped.Shape()->dim(2).dim_value() !=
          gqa_params.num_heads_ * gqa_params.head_size_) {
    LOGS_DEFAULT(WARNING) << "GQA output_reshaped shape not expected";
    return false;
  }
  gqa_params.seq_length_ = output_reshaped.Shape()->dim(1).dim_value();
  return true;
}

// clang-format off
/* Match subgraph of scatter indices calculation
                                                                                        if_prefill (0/1 constant)
                                                                                            |
scatter_indices_left_constant             scatter_indices_right_constant           0 ---> Where <--- Cast <---seqlens_k
              |                                          |                                  |
              |                                         Add <--------------------------- scatter_pos*
              |                                          |
              +--------------------+---------------------+
                                    |
                              scatter_indices
*/
// clang-format on
bool MatchAndCheckScatterIndicesCalculation(
    Graph& graph,
    const Node& scatterND,
    const GQAParameters& gqa_params,
    std::vector<const Node::EdgeEnd*>& result,
    const logging::Logger logger) {
  LOGS_DEFAULT(WARNING) << "Start MatchAndCheckScatterIndicesCalculation";
  // path to seq_length_k
  std::vector<graph_utils::EdgeEndToMatch> scatter_indices_path{
      {0, 1, "Cast", {9, 21}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13, 21}, kOnnxDomain},
      {0, 0, "Concat", {1, 11, 13}, kOnnxDomain},
      {0, 1, "Add", {7, 13, 14}, kOnnxDomain},
      {0, 1, "Where", {9, 16}, kOnnxDomain}};
  if (!graph_utils::FindPath(scatterND, true, scatter_indices_path, result,
                             logger)) {
    LOGS_DEFAULT(WARNING) << "Faild to find scatter indices calculation path";
    return false;
  }

  const Node& cast = result[0]->GetNode();
  const Node& concat = result[2]->GetNode();
  const Node& add = result[3]->GetNode();
  const Node& where = result[4]->GetNode();

  if (!optimizer_utils::IsAttributeWithExpectedValue(cast, "to",
                                                     static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT64))) {
    LOGS_DEFAULT(WARNING) << "Cast attribute to in scatter indices calculation not matched";
    return false;
  }

  if (!optimizer_utils::IsAttributeWithExpectedValue(concat, "axis",
                                                     static_cast<int64_t>(1))) {
    LOGS_DEFAULT(WARNING) << "Concat attribute axis not matched";
    return false;
  }

  const NodeArg& add_input_a = *(add.InputDefs()[0]);
  if (!graph_utils::IsInitializer(graph, add_input_a.Name(), true) ||
      !optimizer_utils::ValidateShape(
          add_input_a, {gqa_params.batch_size_ * gqa_params.seq_length_ *
                            gqa_params.kv_num_heads_,
                        1})) {
    LOGS_DEFAULT(WARNING) << "Add input a not matched";
    return false;
  }

  int32_t where_scatter_idx_input_x_data;
  if (!optimizer_utils::GetScalarInitializerValue(
          graph, *(where.InputDefs()[1]), where_scatter_idx_input_x_data, true) ||
      where_scatter_idx_input_x_data != 0) {
    LOGS_DEFAULT(WARNING) << "Where input x data not matched";
    return false;
  }

  return true;
}

// clang-format off
/* Match subgraph of attention bias calculation
ones_array (shape=B,N,S,P)                                  range_of_qkv_sequence_length_constant (0,1,2,...) (shape=S)
    |                                                                          |
  CumSum (axis=3, exclusive=true, reversed=false)                              Add <--- scatter_pos
    |                                                                          |
    |                                                                        Expand (shape=P,S)
    |                                                                          |
    +-------------------------------> Lesser <------------------------------Transpose (1,0)
                                          |
                                1 ---> Where <--- finfo_min (minimum value of FP32)
                                          |
                                        Cast?
                                          |
                                        Cast?
                                          |
                                    attention_bias
*/
// clang-format on
bool MatchAndCheckAttentionBias(
    Graph& graph,
    const Node& add_before_softmax,
    const GQAParameters& gqa_params,
    const std::vector<const Node::EdgeEnd*>& scatter_edges,
    std::vector<const Node::EdgeEnd*>& result,
    const logging::Logger logger) {
  LOGS_DEFAULT(WARNING) << "Start MatchAndCheckAttentionBias";
  // path to visited 'Where'
  std::vector<graph_utils::EdgeEndToMatch> att_bias_path{
      {0, 1, "Where", {16}, kOnnxDomain},
      {0, 0, "Cast", {9, 21}, kOnnxDomain},
      {0, 0, "Cast", {9, 21}, kOnnxDomain},
      {0, 0, "Less", {13}, kOnnxDomain},
      {0, 1, "Transpose", {21}, kOnnxDomain},
      {0, 0, "Expand", {13}, kOnnxDomain},
      {0, 0, "Add", {14}, kOnnxDomain},
      {0, 1, "Where", {16}, kOnnxDomain}};

  if (!graph_utils::FindPath(add_before_softmax, true, att_bias_path, result,
                             logger)) {
    LOGS_DEFAULT(WARNING) << "Faild to find attention bias calculation path";
    return false;
  }

  const Node& where_att_bias = result[0]->GetNode();
  const Node& cast_0 = result[1]->GetNode();
  const Node& cast_1 = result[2]->GetNode();
  const Node& less = result[3]->GetNode();
  const Node& transpose = result[4]->GetNode();
  const Node& expand = result[5]->GetNode();
  const Node& add = result[6]->GetNode();
  const Node& where_root = result[7]->GetNode();

  if (where_root.Index() != scatter_edges[4]->GetNode().Index()) {
    LOGS_DEFAULT(WARNING) << "where_root in att_bias should be scatter_pos";
    return false;
  }

  if (*(cast_0.OutputDefs()[0]->Type()) != "tensor(bool)") {
    LOGS_DEFAULT(WARNING) << "Cast 0 attribute to in att_bias not matched: " << *(cast_0.OutputDefs()[0]->Type());
    return false;
  }

  if (*(cast_1.OutputDefs()[0]->Type()) != "tensor(uint8)") {
    LOGS_DEFAULT(WARNING) << "Cast 1 attribute to in att_bias not matched: " << *(cast_1.OutputDefs()[0]->Type());
    return false;
  }

  float where_att_bias_input_x_data;
  if (!optimizer_utils::GetScalarInitializerValue(
          graph, *(where_att_bias.InputDefs()[1]), where_att_bias_input_x_data,
          true) ||
      where_att_bias_input_x_data != 1) {
    LOGS_DEFAULT(WARNING) << "Where in att_bias input x data not matched";
    return false;
  }
  // TODO(yuheng): check value of where_att_bias_input_y
  if (!optimizer_utils::ValidateShape(
          *(less.InputDefs()[0]),
          {gqa_params.batch_size_, gqa_params.num_heads_,
           gqa_params.seq_length_, gqa_params.past_seq_length_})) {
    LOGS_DEFAULT(WARNING) << "Less input shape not matched";
    return false;
  }

  InlinedVector<int64_t> perm;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(transpose, "perm", perm) &&
        perm.size() == 2 && perm[0] == 1 && perm[1] == 0)) {
    return false;
  }

  InlinedVector<int64_t> expand_shape_data;
  if (!optimizer_utils::AppendTensorFromInitializer(
          graph, *(expand.InputDefs()[1]), expand_shape_data, true) ||
      expand_shape_data.size() != 2 ||
      expand_shape_data[0] != gqa_params.past_seq_length_ ||
      expand_shape_data[1] != gqa_params.seq_length_) {
    LOGS_DEFAULT(WARNING) << "Expand shape not matched";
    return false;
  }

  if (!optimizer_utils::ValidateShape(*(add.InputDefs()[0]),
                                      {gqa_params.seq_length_})) {
    LOGS_DEFAULT(WARNING) << "Add in att_bias input shape not matched";
    return false;
  }

  return true;
}

bool MatchAndCheckQK(Graph& graph,
                     const Node& add_before_softmax,
                     GQAParameters& gqa_params,
                     const std::vector<const Node::EdgeEnd*>& scatter_edges,
                     std::vector<const Node::EdgeEnd*>& q_edges,
                     std::vector<std::reference_wrapper<const Node>>& present_k_nodes,
                     const logging::Logger logger) {
  LOGS_DEFAULT(WARNING) << "Start match Q*K subgraph";
  // path to input query
  std::vector<graph_utils::EdgeEndToMatch> q_input_path{
      {0, 0, "Mul", {14}, kOnnxDomain},
      {0, 0, "MatMul", {13}, kOnnxDomain},
      {0, 0, "Transpose", {21}, kOnnxDomain},
      {0, 0, "Reshape", {21}, kOnnxDomain}};

  if (!graph_utils::FindPath(add_before_softmax, true, q_input_path, q_edges,
                             logger)) {
    LOGS_DEFAULT(WARNING) << "Faild to find q input path";
    return false;
  }

  const Node& mul = q_edges[0]->GetNode();
  const Node& qk_matmul = q_edges[1]->GetNode();
  const Node& q_transpose = q_edges[2]->GetNode();
  const Node& q_reshape = q_edges[3]->GetNode();

  float scale_data;
  if (!optimizer_utils::GetScalarInitializerValue(graph, *(mul.InputDefs()[1]),
                                                  scale_data, true)) {
    LOGS_DEFAULT(WARNING) << "failed to get scale value";
    return false;
  }
  gqa_params.scale_ = scale_data;

  InlinedVector<int64_t> q_transpose_perm;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(q_transpose, "perm",
                                                    q_transpose_perm) &&
        q_transpose_perm.size() == 4 && q_transpose_perm[0] == 0 &&
        q_transpose_perm[1] == 2 && q_transpose_perm[2] == 1 &&
        q_transpose_perm[3] == 3)) {
    return false;
  }

  if (!ValidateReshapeShape(graph, *(q_reshape.InputDefs()[1]),
                            {gqa_params.batch_size_, gqa_params.seq_length_,
                             gqa_params.num_heads_, gqa_params.head_size_})) {
    LOGS_DEFAULT(WARNING) << "q_reshape shape not matched";
    return false;
  }

  // path to input key
  const Node* maybe_k_transpose = graph_utils::GetInputNode(qk_matmul, 1);
  if (maybe_k_transpose == nullptr || maybe_k_transpose->OpType().compare("Transpose") != 0) {
    LOGS_DEFAULT(WARNING) << "qk_matmul input[1] mismatch";
    return false;
  }

  const Node* k_transpose_input = graph_utils::GetInputNode(*maybe_k_transpose, 0);
  if (k_transpose_input == nullptr) {
    LOGS_DEFAULT(WARNING) << "empty k_transpose input";
    return false;
  }

  std::vector<std::reference_wrapper<const Node>> present_k_scatternd_nodes;
  MatchKVExpand(k_transpose_input, present_k_scatternd_nodes, logger);
  if (present_k_scatternd_nodes.empty()) {
    LOGS_DEFAULT(WARNING) << "Failed to find present_k expand path";
    return false;
  }

  const Node& scatterND_k = present_k_scatternd_nodes.back();
  const Node* maybe_k_reshape = graph_utils::GetInputNode(scatterND_k, 2);
  if (maybe_k_reshape == nullptr || maybe_k_reshape->OpType().compare("Reshape") != 0) {
    LOGS_DEFAULT(WARNING) << "scatterND_k input[2] mismatch";
    return false;
  }

  if (graph_utils::GetInputNode(scatterND_k, 1)->Index() !=
      scatter_edges[0]->GetNode().Index()) {
    LOGS_DEFAULT(WARNING) << "scatterND_k input 1 mismatch, expected: "
                          << scatter_edges[0]->GetNode().Index() << " got: "
                          << graph_utils::GetInputNode(scatterND_k, 1)->Index();
    return false;
  }

  if (!optimizer_utils::ValidateShape(
          *(scatterND_k.InputDefs()[0]),
          {gqa_params.batch_size_, gqa_params.kv_num_heads_,
           gqa_params.past_seq_length_, gqa_params.head_size_})) {
    LOGS_DEFAULT(WARNING) << "scatterND_k input 0 shape not matched";
    return false;
  }

  if (!ValidateReshapeShape(graph, *(maybe_k_reshape->InputDefs()[1]),
                            {gqa_params.batch_size_, gqa_params.seq_length_,
                             gqa_params.kv_num_heads_, gqa_params.head_size_})) {
    LOGS_DEFAULT(WARNING) << "K_reshape shape not matched";
    return false;
  }

  InlinedVector<int64_t> k_transpose_perm;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(*maybe_k_transpose, "perm",
                                                    k_transpose_perm) &&
        k_transpose_perm.size() == 4 && k_transpose_perm[0] == 0 &&
        k_transpose_perm[1] == 1 && k_transpose_perm[2] == 3 &&
        k_transpose_perm[3] == 2)) {
    LOGS_DEFAULT(WARNING) << "K_transpose perm not matched";
    return false;
  }

  present_k_nodes.push_back(*maybe_k_transpose);
  present_k_nodes.insert(present_k_nodes.end(), present_k_scatternd_nodes.begin(),
                         present_k_scatternd_nodes.end());
  present_k_nodes.push_back(*maybe_k_reshape);

  return true;
}
}  // namespace

Status GroupQueryAttentionFusion::ApplyImpl(
    Graph& graph,
    bool& modified,
    int graph_level,
    const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  int fuse_count = 0;
  for (auto node_idx : node_topology_list) {
    auto node_ptr = graph.GetNode(node_idx);
    if (node_ptr == nullptr)
      continue;

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // find the second matmul which inputs are from softmax and scatterND
    if (graph_utils::IsSupportedProvider(node,
                                         GetCompatibleExecutionProviders()) &&
        graph_utils::IsSupportedOptypeVersionAndDomain(
            node, "MatMul", {1, 9, 13}, kOnnxDomain)) {
      const Node* matmul_input_0 = graph_utils::GetInputNode(node, 0);
      const Node* matmul_input_1 = graph_utils::GetInputNode(node, 1);
      if (matmul_input_0 == nullptr ||
          matmul_input_0->OpType().compare("Softmax") != 0) {
        continue;
      }

      if (matmul_input_1 == nullptr) {
        continue;
      }
      std::vector<std::reference_wrapper<const Node>> present_v_nodes;
      MatchKVExpand(matmul_input_1, present_v_nodes, logger);
      if (present_v_nodes.empty()) {
        continue;
      }

      const NodeArg& matmul_input_1_shape = *(matmul_input_1->OutputDefs()[0]);
      if (!optimizer_utils::IsShapeKnownOnAllDims(matmul_input_1_shape, 4)) {
        continue;
      }
      GQAParameters gqa_params;
      gqa_params.batch_size_ = matmul_input_1_shape.Shape()->dim(0).dim_value();
      gqa_params.num_heads_ = matmul_input_1_shape.Shape()->dim(1).dim_value();
      gqa_params.past_seq_length_ =
          matmul_input_1_shape.Shape()->dim(2).dim_value();
      gqa_params.head_size_ = matmul_input_1_shape.Shape()->dim(3).dim_value();

      if (GroupQueryAttentionFusion::FuseSubGraph(graph, node, *matmul_input_0,
                                                  present_v_nodes, gqa_params,
                                                  logger)) {
        fuse_count++;
        modified = true;
      }
    }
  }

  LOGS_DEFAULT(WARNING) << "Total fused GroupQueryAttention node count: "
                        << fuse_count;

  return Status::OK();
}

// static
// clang-format off
/** Fuse GroupQueryAttention SubGraph.
 Abbreviatios: B is batch_size, S is sequence_length, W is hidden_size, P is past_sequence_length
               N is number of attention heads, H is head size, and W=N*H, h=Sqrt(H)
               B and S could be symbolic. ? means it is optional.
    GQA inputs: query, key value, past_key, past_value, seqlens_k, total_sequence_length
    layout: q, k, v: (B, S, W), past_k, past_v: (B, N, S, H), seqlens_k: (B), total_sequence_length: (1)

          query             key       value
            |                |          |
         Reshape          Reshape     Reshape (B,S,H,N)
            |      past_key  /          |
            |         |     /           |
          q_Transpose |    /            |
           (0,2,1,3)  |   /             |                    seqlens_k
             \        |  /              |                     /     |
              \       | /               |                    /      |
present_key<---\----ScatterND <---------|-----(scatter_indices*)    |
               |      |                 |        /                  |
               |  k_Transpose           |       /                   |
               \  (0,1,3,2)             |      /                    |
                \     |                 |     /                     |
                 \  Expand(G)           |    /                      |
                  \   |     past_value  |   /                       |
                qk_MatMul          \    |  /                        |
                      |            ScatterND-----> present_value    |
                  qk_Mul              /                             |
                      |              /                              |
                     Add <----------/--------(attention_bias, one/finfo_min mask*)
                      |            /
                    Softmax     Expand(G)
                       \         /
                        \       /
                      qkv_MatMul
                             |
                          Transpose (perm=0,2,1,3)
                             |
                          Reshape---(shape=B,S,W)
                             |
                           output

After Fusion:
 [q] [k] [v] [past_k] [past_v] [seqlens_k] [total_seq_len]
  |   |   |    |        |        |          |
  \   |   |    |        /        /          /
   \  \   |    |      /         /         /
          GroupQueryAttention
          /         |       \
    present_k    output     present_v
*/
// clang-format on
bool GroupQueryAttentionFusion::FuseSubGraph(
    Graph& graph,
    const Node& qkv_matmul,
    const Node& softmax,
    std::vector<std::reference_wrapper<const Node>>& present_v_nodes,
    GQAParameters& gqa_params,
    const logging::Logger& logger) {
  // path to output
  std::vector<graph_utils::EdgeEndToMatch> output_path{
      {0, 0, "Transpose", {1, 13, 21}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13, 21}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> output_edges;
  if (!graph_utils::FindPath(qkv_matmul, false, output_path, output_edges,
                             logger)) {
    LOGS_DEFAULT(WARNING) << "Faild to find output path";
    return false;
  }

  if (!CheckNodesInOutputPath(graph, output_edges[1]->GetNode(),
                              output_edges[0]->GetNode(), gqa_params)) {
    return false;
  }

  // check past_value and input value
  const Node& scatterND_v = present_v_nodes.back();
  const Node* maybe_reshape = graph_utils::GetInputNode(scatterND_v, 2);
  if (maybe_reshape == nullptr ||
      maybe_reshape->OpType().compare("Reshape") != 0) {
    return false;
  }

  const NodeArg& input_value = *(maybe_reshape->InputDefs()[0]);
  if (!optimizer_utils::ValidateShape(
          input_value, {gqa_params.batch_size_, gqa_params.seq_length_, -1})) {
    LOGS_DEFAULT(WARNING) << "input_value size mismatch";
    return false;
  }
  if (input_value.Shape()->dim(2).dim_value() % gqa_params.head_size_ != 0) {
    LOGS_DEFAULT(WARNING) << "input_value size mismatch";
    return false;
  }
  gqa_params.kv_num_heads_ =
      input_value.Shape()->dim(2).dim_value() / gqa_params.head_size_;

  const NodeArg& past_value = *(scatterND_v.InputDefs()[0]);
  if (!optimizer_utils::ValidateShape(
          past_value, {gqa_params.batch_size_, gqa_params.kv_num_heads_,
                       gqa_params.past_seq_length_, gqa_params.head_size_})) {
    LOGS_DEFAULT(WARNING) << "past_value shape not matched";
    return false;
  }

  // match calculation of scatterND indices subgraph
  std::vector<const Node::EdgeEnd*> scatter_indices_edges;
  if (!MatchAndCheckScatterIndicesCalculation(graph, scatterND_v, gqa_params,
                                              scatter_indices_edges, logger)) {
    LOGS_DEFAULT(WARNING) << "failed to match scatter indices calculation";
    return false;
  }

  if (softmax.OpType().compare("Softmax") != 0) {
    LOGS_DEFAULT(WARNING) << "wrong matmul input softmax";
    return false;
  }

  const Node* maybe_add_before_softmax = graph_utils::GetInputNode(softmax, 0);
  if (maybe_add_before_softmax == nullptr ||
      maybe_add_before_softmax->OpType().compare("Add") != 0) {
    return false;
  }
  // match attention bias subgraph
  std::vector<const Node::EdgeEnd*> attention_bias_edges;
  if (!MatchAndCheckAttentionBias(graph, *maybe_add_before_softmax, gqa_params,
                                  scatter_indices_edges, attention_bias_edges,
                                  logger)) {
    LOGS_DEFAULT(WARNING) << "failed to match attention bias subgraph";
    return false;
  }

  // match QK
  std::vector<const Node::EdgeEnd*> query_input_edges;
  std::vector<std::reference_wrapper<const Node>> present_k_nodes;
  if (!MatchAndCheckQK(graph, *maybe_add_before_softmax, gqa_params,
                       scatter_indices_edges, query_input_edges,
                       present_k_nodes, logger)) {
    LOGS_DEFAULT(WARNING) << "failed to match Q*K subgraph";
    return false;
  }

  // create GQA node
  // input list: [query, key, value, past_key, past_value, seqlens_k,
  // total_seq_len]
  ONNX_NAMESPACE::TensorProto total_seq_length;
  total_seq_length.set_name("total_seq_len");
  total_seq_length.add_dims(1);
  total_seq_length.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  total_seq_length.add_int32_data(
      static_cast<int32_t>(gqa_params.past_seq_length_));
  graph.AddInitializedTensor(total_seq_length);
  NodeArg* total_seq_len_node_arg =
      &graph.GetOrCreateNodeArg(total_seq_length.name(), nullptr);

  const Node& scatterND_k = present_k_nodes[present_k_nodes.size() - 2].get();
  const std::array input_defs{
      graph.GetNode(query_input_edges[3]->GetNode().Index())->MutableInputDefs()[0],
      graph.GetNode(present_k_nodes.back().get().Index())->MutableInputDefs()[0],
      graph.GetNode(maybe_reshape->Index())->MutableInputDefs()[0],
      graph.GetNode(scatterND_k.Index())->MutableInputDefs()[0],
      graph.GetNode(scatterND_v.Index())->MutableInputDefs()[0],
      graph.GetNode(scatter_indices_edges[4]->GetNode().Index())->MutableInputDefs()[2],
      total_seq_len_node_arg};
  // output list: [output, present_key, present_value]
  const std::array output_defs{
      graph.GetNode(output_edges[1]->GetNode().Index())->MutableOutputDefs()[0],
      graph.GetNode(scatterND_k.Index())
          ->MutableOutputDefs()[0],
      graph.GetNode(scatterND_v.Index())->MutableOutputDefs()[0]};
  Node& gqa_node = graph.AddNode("GroupQueryAttention", "GroupQueryAttention",
                                 "Fused GroupQueryAttention subgraphs",
                                 input_defs, output_defs, nullptr, kMSDomain);
  gqa_node.AddAttribute("num_heads", gqa_params.num_heads_);
  gqa_node.AddAttribute("kv_num_heads", gqa_params.kv_num_heads_);
  gqa_node.AddAttribute("scale", gqa_params.scale_);
  gqa_node.SetExecutionProviderType(qkv_matmul.GetExecutionProviderType());

  // remove fused nodes
  std::set<NodeIndex> nodes_to_remove{
      qkv_matmul.Index(),
      softmax.Index(),
      maybe_reshape->Index(),
      maybe_add_before_softmax->Index(),
  };
  std::transform(
      present_v_nodes.begin(), present_v_nodes.end(),
      std::inserter(nodes_to_remove, nodes_to_remove.end()),
      [](std::reference_wrapper<const Node> node_ref_wrapper) -> NodeIndex {
        return node_ref_wrapper.get().Index();
      });
  std::transform(
      present_k_nodes.begin(), present_k_nodes.end(),
      std::inserter(nodes_to_remove, nodes_to_remove.end()),
      [](std::reference_wrapper<const Node> node_ref_wrapper) -> NodeIndex {
        return node_ref_wrapper.get().Index();
      });
  auto append_to_remove_list_from_edge =
      [&](std::vector<const Node::EdgeEnd*> edges) {
        std::transform(
            edges.begin(), edges.end(),
            std::inserter(nodes_to_remove, nodes_to_remove.end()),
            [](const Node::EdgeEnd* edge) { return edge->GetNode().Index(); });
      };
  append_to_remove_list_from_edge(output_edges);
  append_to_remove_list_from_edge(scatter_indices_edges);
  append_to_remove_list_from_edge(attention_bias_edges);
  append_to_remove_list_from_edge(query_input_edges);

  LOGS_DEFAULT(WARNING) << "nodes_to_remove set size: " << nodes_to_remove.size();

  for (const auto& node_index : nodes_to_remove) {
    Node* node = graph.GetNode(node_index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  return true;
}

}  // namespace onnxruntime
