// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "graph/attr_value.h"
#include "graph/op/all_ops.h"
#include "graph/op/nn_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"

namespace TNN_NS {

DECLARE_NPU_LAYER_WEIGHT(Reshape, LAYER_RESHAPE)

Status NpuReshapeLayer::Convert() {
    auto param = dynamic_cast<ReshapeLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    // shape
    std::shared_ptr<hiai::op::Const> shape_const = std::make_shared<hiai::op::Const>(layer_name_ + "_input_size");
    int shape_count                              = param->shape.size();
    hiai::TensorDesc desc(hiai::Shape({shape_count}), hiai::FORMAT_NCHW, hiai::DT_INT32);
    NpuUtils::CreateAttrArray(shape_const, param->shape, desc, shape_count);
    weight_ops_.push_back(shape_const);

    auto output = std::make_shared<hiai::op::Reshape>(outputs_name_[0]);
    output->set_input_x(*input_ops_[0]->GetOperator());
    output->set_input_shape(*shape_const);
    output->set_attr_axis(param->axis);
    output->set_attr_num_axes(param->num_axes);
    ADD_OUTPUT_OP(output)
}

REGISTER_NPU_LAYER(Reshape, LAYER_RESHAPE)

}  // namespace TNN_NS
