#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <pybind11/eigen.h>
#include <tuple>
#include "ceres/rotation.h"
#include "math.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

namespace py = pybind11;
const int camera_len {6};

// Templated so that we can use Ceres's automatic differentiation to compute analytic jacobians.
struct ReprojectionError {
ReprojectionError(double observed_x, double observed_y, Eigen::Matrix<double, 3, 3> K)
    : observed_x(observed_x), observed_y(observed_y), K_(K) {}

template <typename T>
bool operator()(const T* const camera,
                const T* const point,
                T* residuals) const {   
    
    // Angle axis
    T p[3];
    
    ceres::AngleAxisRotatePoint(camera, point, p);

    // Create Eigen matrix of the rotated point
    Eigen::Map<Eigen::Matrix<T, 3, 1>> trans_point(p);

    // Create Eigen matrix for translation
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(camera+3);

    // Apply translation
    trans_point(0,0) += t(0,0);
    trans_point(1,0) += t(1,0);
    trans_point(2,0) += t(2,0);

    // Perspective divide
    T xp = trans_point(0,0) / trans_point(2,0); 
    T yp = trans_point(1,0) / trans_point(2,0);

    // Apply focal length and principal point
    T predicted_x = T(K_(0,0)) * xp + T(K_(0,2));
    T predicted_y = T(K_(1,1)) * yp + T(K_(1,2)); 
    
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);
    return true;
}

// Factory to hide the construction of the CostFunction object from
// the client code.
static ceres::CostFunction* Create(const double observed_x,
                                    const double observed_y, Eigen::Matrix<double,3,3> K) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, camera_len, 3>(
                new ReprojectionError(observed_x, observed_y, K)));
}

double observed_x;
double observed_y;
Eigen::Matrix<double, 3, 3> K_;
};

// Function to call from Python
void ceres_BA(int num_cameras, int num_points, int num_observations, py::array_t<double> parameters, py::array_t<double> observations, py::array_t<int>  camera_index, py::array_t<int>  point_index, Eigen::Matrix<double, 3, 3> K){

    // double *params is a pointer to first camera
    py::buffer_info param_buf = parameters.request();
    double *params = (double *) param_buf.ptr;

    py::buffer_info observ_buf = observations.request();
    const double *observs = (double *) observ_buf.ptr;

    py::buffer_info camera_idx_buf = camera_index.request();
    int *camera_idx = (int *) camera_idx_buf.ptr;

    py::buffer_info point_idx_buf = point_index.request();
    int *point_idx = (int *) point_idx_buf.ptr;

    // Pointer to the first pos of the 3d points
    double* points_start = params + camera_len * num_cameras;
   
    ceres::Problem problem;

    // Sets first camera constant
    problem.AddParameterBlock(params, camera_len);
    problem.SetParameterBlockConstant(params);

    for (int i = 0; i < num_observations; ++i) {

        double* mutable_camera_for_observation = params + camera_idx[i] * camera_len;

        double* mutable_point_for_observation = points_start + point_idx[i] * 3;
  
        ceres::CostFunction* cost_function =
            ReprojectionError::Create( observs[2 * i + 0], observs[2 * i + 1], K);

        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), mutable_camera_for_observation, mutable_point_for_observation);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

// Binds the functions and classes so it can be used as a module in Python
PYBIND11_MODULE(ceres_python, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: ceres_python

        .. autosummary::
           :toctree: _generate

           ceres_BA

    )pbdoc";

    m.def("ceres_BA", &ceres_BA, R"pbdoc(
        Function to call from Python to run the Bundle Adjustment
        
    )pbdoc");
    
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}