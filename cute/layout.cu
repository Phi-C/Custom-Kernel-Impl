/*
 * Usage: nvcc -o layout layout.cu -arch=sm_80 -std=c++17
 * -I/path/to/cutlass/include
 */

#include <cublas_v2.h>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>
#include <typeinfo>

using T = float;
using namespace cute;

// Implementation of cute::print_layout
template <class Shape, class Stride>
void print2D(Layout<Shape, Stride> const& layout) {
    for (int m = 0; m < size<0>(layout); ++m) {
        for (int n = 0; n < size<1>(layout); ++n) {
            printf("%3d  ", layout(m, n));
        }
        printf("\n");
    }
}

int main() {
    /**
     * Layout abstraction
     */
    // Case 1:
    cute::Layout layout1 =
        cute::make_layout(cute::make_shape(4, 6), cute::make_stride(6, 1));
    cute::print_layout(layout1);

    // Case 2:
    cute::Layout a = cute::Layout<_3, _1>{};
    cute::Layout b = cute::Layout<_4, _3>{};
    cute::Layout layout2 = cute::make_layout(a, b);
    cute::print_layout(layout2);

    // Case 3: composition case
    // cute::Layout<shape, stride>{}
    // cute::make_layout(layout_a, layout_b) ---> new layout's shape = (a.shape,
    // b.shape), new layout's stride = (a.stride, b.stride)
    cute::Layout c = cute::Layout<Shape<_2, _4>, Shape<_3, _6>>{};
    cute::Layout d = cute::Layout<Shape<_3, _5>, Shape<_1, _24>>{};
    cute::Layout layout3 = cute::make_layout(c, d);
    cute::print_layout(layout3);
    // print2D(layout3);

    // Case 4: use coordinate to slice
    // auto coord_start = cutlass::make_Coord((_, 1), (_, 1));
    // auto coord_end = cutlass::make_Coord((_, 2), (_, 2));
    auto row_coord = cutlass::make_Coord(0);
    auto col_coord = cutlass::make_Coord(0);
    auto coord = cutlass::make_Coord(row_coord, col_coord);
    cute::Layout layout_out = cute::slice(coord, layout3);
    auto shape_out = layout_out.shape();
    auto stride_out = layout_out.stride();
    std::cout << typeid(layout2).name() << std::endl;
    std::cout << typeid(layout_out).name() << std::endl;
    std::cout << "shape_out: " << shape_out << std::endl;
    std::cout << "stride_out: " << stride_out << std::endl;
    // It seems layout_out is dynamic, and cannot be printed.

    /*
     * MMA abstraction
     */
    TiledMMA mma1 = make_tiled_mma(SM80_8x8x4_F64F64F64F64_TN{});

    print_latex(mma1);

    // base case
    TiledMMA mma2 = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{});
    print_latex(mma2);

    // Repeat MMA_Atomic via AtomicLayoutMNK, more threads involed
    TiledMMA mma3 = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                                   Layout<Shape<_2, _2, _1>>{});
    print_latex(mma3);

    // Repeat MMA_Atomic via ValLayoutMNK, equivalent to previous version's
    // TiledMMA mma4 = make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
    //                             Layout<Shape<_1, _1, _1>>{},
    //                             Layout<_1, _2, _1>{});
    TiledMMA mma4 =
        make_tiled_mma(SM80_16x8x16_F16F16F16F16_TN{},
                       Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
    print_latex(mma4);

    return 0;
}
