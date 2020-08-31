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

DECLARE_NPU_LAYER(DetectionOutput, LAYER_DETECTION_OUTPUT)

Status NpuDetectionOutputLayer::Convert() {
    // parameter and weight of the DetectionOutput layer
    auto param = dynamic_cast<DetectionOutputLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    std::shared_ptr<hiai::op::Data> boxNum       = std::make_shared<hiai::op::Data>(outputs_name_[0]);
    std::shared_ptr<hiai::op::Data> regionalProp = std::make_shared<hiai::op::Data>(outputs_name_[0]);

    // TNN params: 0 loc  1  conf  2 prior box
    auto output = std::make_shared<hiai::op::SSDDetectionOutput>(outputs_name_[0]);
    output->set_input_mbox_conf(*input_ops_[1]->GetOperator());
    output->set_input_mbox_loc(*input_ops_[0]->GetOperator());
    output->set_input_mbox_priorbox(*input_ops_[2]->GetOperator());
    output->set_attr_num_classes(param->num_classes);
    output->set_attr_shared_location(param->share_location);
    output->set_attr_background_label_id(param->background_label_id);
    output->set_attr_nms_threshold(param->nms_param.nms_threshold);
    output->set_attr_num_classes(param->nms_param.top_k);
    output->set_attr_num_classes(param->eta);
    output->set_attr_variance_encoded_in_target(param->variance_encoded_in_target);
    output->set_attr_code_type(param->code_type);
    output->set_attr_keep_top_k(param->keep_top_k);
    output->set_attr_confidence_threshold(param->confidence_threshold);

    ADD_OUTPUT_OP(output)
}
REGISTER_NPU_LAYER(DetectionOutput, LAYER_DETECTION_OUTPUT)

}  // namespace TNN_NS