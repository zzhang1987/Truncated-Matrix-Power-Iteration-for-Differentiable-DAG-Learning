
#include <torch/extension.h>
#include <iostream>
#include <cmath>

template <typename T>
std::tuple<torch::Tensor, torch::Tensor>  h_tmpi(torch::Tensor B, T eps) {
    auto d = B.sizes()[0];
    auto _g = torch::clone(B);
    auto _B = torch::clone(B);
    auto _grad = torch::matrix_power(B, 0);

    auto old_g = torch::zeros_like(B);
    auto old_B = torch::zeros_like(B);
    auto sec_g = torch::zeros_like(B);
    
    int j = 1;

    while(2*j <= 2 * d){
        old_B.copy_(_B);
        old_g.copy_(_g);

        at::matmul_out(_g, _B, _g);
        at::add_out(_g, old_g, _g);

        sec_g.copy_(_grad);
        at::matmul_out(_grad, _B.t(), _grad);
        at::matmul_out(_B, _B, _B);
        at::add_out(_grad, _grad, sec_g); 
        sec_g.copy_(_g);

        at::add_out(sec_g, sec_g, old_g, -1.0);
        at::add_out(sec_g, sec_g, old_B);
        at::add_out(sec_g, sec_g, _B, -1.0);
        sec_g = at::mul(sec_g,  at::Scalar(static_cast<T>(j)));

        at::add_out(_grad, _grad, sec_g.t());

        if (at::max(_B).item().to<T>() < eps){
            break;
        }
        j *= 2;            
    }
    return {torch::trace(_g), _grad};
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h_tmpi_native_double", &h_tmpi<double>, "H_tmpi native run");
  m.def("h_tmpi_native_float", &h_tmpi<float>, "H_tmpi native run");
}
