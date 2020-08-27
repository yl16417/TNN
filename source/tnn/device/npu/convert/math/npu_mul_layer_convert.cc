#include <graph/op/all_ops.h>
#include "npu_binary_layer_convert.h"
#include "tnn/device/npu/convert/npu_base_layer_convert.h"
#include "tnn/device/npu/convert/npu_utils.h"

namespace TNN_NS {

class NpuMulLayer : public NpuBinaryLayer {
public:
    NpuMulLayer(LayerType ignore) : NpuBinaryLayer(LAYER_MUL) {}
    ~NpuMulLayer() {}

protected:
    Status Convert() {
        return NpuBinaryLayer::BinaryConvert<hiai::op::Mul>();
    }
};

REGISTER_NPU_LAYER(Mul, LAYER_MUL);

}  // namespace TNN_NS