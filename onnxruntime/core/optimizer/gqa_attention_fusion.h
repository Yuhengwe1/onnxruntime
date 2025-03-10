// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

struct GQAParameters;

/**
@Class AttentionFusion
Rewrite graph fusing attention subgraph to a single Attention node.
*/
class GroupQueryAttentionFusion : public GraphTransformer {
 public:
  GroupQueryAttentionFusion(const InlinedHashSet<std::string_view>&
                                compatible_execution_providers = {}) noexcept
      : GraphTransformer("GroupQueryAttentionFusion",
                         compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph,
                   bool& modified,
                   int graph_level,
                   const logging::Logger& logger) const override;

 private:
  static bool FuseSubGraph(
      Graph& graph,
      const Node& qkv_matmul,
      const Node& softmax,
      std::vector<std::reference_wrapper<const Node>>& present_v_nodes,
      GQAParameters& gqa_params,
      const logging::Logger& logger);
};

}  // namespace onnxruntime
