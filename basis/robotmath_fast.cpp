#include <iostream>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

//https://stackoverflow.com/questions/49582252/pybind-numpy-access-2d-nd-arrays
py::array_t<double> rotmat_from_axangle(py::array_t<double> axis, double angle) {
    py::buffer_info axis_buf = axis.request();
    if (axis_buf.size != 3) {
        throw std::runtime_error("Input shapes must be 3");
    } 
    py::array_t<double> result = py::array_t<double>(9);
    py::buffer_info result_buf = result.request();
    
    // data pointer
    double* axis_ptr = (double*)axis_buf.ptr,
          * result_ptr = (double*)result_buf.ptr;
    
    
    // operation
    double a = std::cos(angle/2.0);
    double b=-axis_ptr[0] * std::sin(angle / 2.0),
           c= -axis_ptr[1] * std::sin(angle / 2.0),
           d = -axis_ptr[2] * std::sin(angle / 2.0);
    double aa = a * a,
        bb = b * b,
        cc = c * c,
        dd = d * d,
        bc = b * c,
        ad = a * d, 
        ac = a * c,
        ab = a * b, 
        bd = b * d,
        cd = c * d; 
    result_ptr[0] = aa + bb - cc - dd;
    result_ptr[1] = 2.0 * (bc + ad);
    result_ptr[2] = 2.0 * (bd - ac);
    result_ptr[3] = 2.0 * (bc - ad);
    result_ptr[4] = aa + cc - bb - dd;
    result_ptr[5] = 2.0 * (cd + ab);
    result_ptr[6] = 2.0 * (bd + ac);
    result_ptr[7] = 2.0 * (cd - ab);
    result_ptr[8] = aa + dd - bb - cc; 

    //// reshape array to match input shape
    result.resize({ 3,3 });

    return result;
}

py::array_t<double> unit_vector(py::array_t<double> vector) {
    py::buffer_info vector_buf = vector.request();

    py::array_t<double> result = py::array_t<double>(vector_buf.size);
    py::buffer_info result_buf = result.request();

    // data pointer
    double* vector_ptr = (double*)vector_buf.ptr,
        * result_ptr = (double*)result_buf.ptr;


    double length = 0;
    for (int i = 0; i < vector_buf.size; i++)
        length += std::pow(vector_ptr[i], 2);
    length = std::sqrt(length);

    if (length < 1e-9){
        for (int i = 0; i < result_buf.size; i++)
            result_ptr[i]=0;
    }
    else {
        for (int i = 0; i < result_buf.size; i++)
            result_ptr[i] = vector_ptr[i]/length;
    }

    return result;
}




PYBIND11_MODULE(robotmath_fast, m) {
    m.doc() = "robot math using pybind11"; // optional module docstring 
    m.def("rotmat_from_axangle", &rotmat_from_axangle, "Compute the rodrigues matrix using the given axis and angle");
    m.def("unit_vector", &unit_vector, "Normalize a vector");
}