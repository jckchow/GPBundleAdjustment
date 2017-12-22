#include "ceres/ceres.h"
#include "ceres/cost_function.h"
#include "ceres/cubic_interpolation.h"
#include "glog/logging.h"
#include "Python.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <Eigen/Dense>

// Define constants
#define PI 3.141592653589793238462643383279502884197169399

#define ARRAYSIZE(a) sizeof(a) / sizeof(a[0]);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Pseudo observation of a constant
/// Input:    y      - Some constant value
///           weight - 1 / StdDev
/// Unknowns: x      - some unknown parameter in the adjustment
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct constantConstraint {
  
  constantConstraint(double y, double weight)
        : y_(y), weight_(weight)  {}

  template <typename T>
  // unknown parameters followed by the output residual
  bool operator()(const T* const x, T* residual) const {

  residual[0] = x[0] - T(y_);
  residual[0] *= T(weight_);
  return true;
  }

 private:
  // Observations for a sample.
  const double y_;
  const double weight_;
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// f(x,y) = (1-x)^2 + 100(y - x^2)^2;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Rosenbrock : public ceres::FirstOrderFunction {
 public:
  virtual ~Rosenbrock() {}
  virtual bool Evaluate(const double* parameters,
                        double* cost,
                        double* gradient) const {
    const double x = parameters[0];
    const double y = parameters[1];
    cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
    if (gradient != NULL) {
      gradient[0] = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
      gradient[1] = 200.0 * (y - x * x);
    }
    return true;
  }
  virtual int NumParameters() const { return 2; }
};

// class MLEGaussian : public ceres::FirstOrderFunction {
//  public:
//   virtual ~MLEGaussian() {}

//   // Input parameters for this class
//   std::vector<double> y; // vector of observations that we pass in
//   std::vector<double> stdDev; // standard deviation of the obs
//   int u; // number of unknowns

//   virtual bool Evaluate(const double* parameters, // array of unknowns
//                         double* cost,
//                         double* gradient) const {
//     int n = y.size();
//     // std::cout<<"n: "<<n<<std::endl;
//     // std::cout<<"u: "<<u<<std::endl;
//     Eigen::MatrixXd A(n,u);
//     Eigen::MatrixXd X(u,1);
//     Eigen::MatrixXd v(n,1);
//     Eigen::MatrixXd w(n,1);
//     Eigen::MatrixXd Cl(n,n);
//     for (int i = 0; i < u; i++)
//     {
//         X(i,0) = parameters[i];
//     }

//     Cl.setZero();
//     for (int i = 0; i < n; i++)
//     {
//         Cl(i,i) = stdDev[i]*stdDev[i];
//         w(i,0) = parameters[i] - y[i];
//     }
//     v = X + w;

//     Eigen::MatrixXd f =  0.5 * v.transpose() * Cl.inverse() * v;
//     // std::cout<<"f"<<std::endl;
//     // std::cout<<f<<std::endl;
//     cost[0] = f(0,0);
//     if (gradient != NULL) {
//       Eigen::MatrixXd jacobian = Cl.inverse() * w;
//     //   std::cout<<"gradient"<<std::endl;
//     //   std::cout<<jacobian<<std::endl;
//     for (int i = 0; i < n; i++)
//     {
//       gradient[i] = jacobian(i,0);
//     }
//     //   std::cout<<"gradient0: "<<gradient[0]<<std::endl;
//     //   std::cout<<"gradient1: "<<gradient[1]<<std::endl;
//     }
//     return true;
//   }
//   virtual int NumParameters() const { return u; }
// };



class MLEGaussian : public ceres::FirstOrderFunction {
 public:
  virtual ~MLEGaussian() {}

  // Input parameters for this class
  std::vector<double> y; // vector of observations that we pass in
  std::vector<double> stdDev; // standard deviation of the obs
  int u; // number of unknowns

  virtual bool Evaluate(const double* parameters, // array of unknowns
                        double* cost,
                        double* gradient) const {
    int n = y.size();

    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

    std::cout<<"y_mean: "<< y_mean<<std::endl;

    std::cout<<"n: "<<n<<std::endl;
    std::cout<<"u: "<<u<<std::endl;
    Eigen::MatrixXd A(n,u);
    Eigen::MatrixXd X(u,1);
    Eigen::MatrixXd v(n,1);
    Eigen::MatrixXd w(n,1);
    Eigen::MatrixXd Cl(n,n);
    for (int i = 0; i < u; i++)
    {
        X(i,0) = parameters[i];
    }

    Cl.setZero();
    for (int i = 0; i < n; i++)
    {
        Cl(i,i) = stdDev[i]*stdDev[i];
        w(i,0) = parameters[i] - y[i]; // fx - l
    }
    v = X + w;

    Eigen::MatrixXd f =  0.5 * v.transpose() * Cl.inverse() * v;
    // std::cout<<"f"<<std::endl;
    // std::cout<<f<<std::endl;
    cost[0] = f(0,0);
    if (gradient != NULL) {
      Eigen::MatrixXd jacobian = Cl.inverse() * w;
    //   std::cout<<"gradient"<<std::endl;
    //   std::cout<<jacobian<<std::endl;
    for (int i = 0; i < n; i++)
    {
      gradient[i] = jacobian(i,0);
    }
    //   std::cout<<"gradient0: "<<gradient[0]<<std::endl;
    //   std::cout<<"gradient1: "<<gradient[1]<<std::endl;
    }
    return true;
  }
  virtual int NumParameters() const { return u; }
};







/////////////////////////
/// MAIN CERES-SOLVER ///
/////////////////////////
int main(int argc, char** argv) {
    Py_Initialize();
    google::InitGoogleLogging(argv[0]);

    PyRun_SimpleString("import matplotlib.pyplot as plt");
    PyRun_SimpleString("import numpy as np");
    PyRun_SimpleString("import time as TIME");
    PyRun_SimpleString("t0 = TIME.clock()");        

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Initialize the unknowns
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////   
    std::vector<double> X;
    X.push_back(0.55*-3);
    X.push_back(0.55*-2);
    X.push_back(0.55*-0.6);
    X.push_back(0.55*0.4);
    X.push_back(0.55*1.0);
    X.push_back(0.55*1.6);


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Set up cost functions
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    PyRun_SimpleString("t0 = TIME.clock()");        
    PyRun_SimpleString("print 'Start building Ceres-Solver cost functions' ");     
   
    // ceres::Problem problem;
    // ceres::LossFunction* loss = NULL;
    // loss = new ceres::HuberLoss(1.0);
    // loss = new ceres::CauchyLoss(0.5);

    // if(true)
    // {
    //     for(int n = 0; n < X.size(); n++)
    //     {
    //         ceres::CostFunction* cost_function =
    //             new ceres::AutoDiffCostFunction<constantConstraint, 1, 1>(
    //                 new constantConstraint(0, 1.0/1E-4));
    //         problem.AddResidualBlock(cost_function, NULL, &X[n]);
    //     }
    // }

    for(int n = 0; n < X.size(); n++)
        std::cout<<"before X: "<<X[n]<<std::endl;


    MLEGaussian* leastSquaresProblem = new MLEGaussian;
    leastSquaresProblem->y.push_back(0.55*-3);
    leastSquaresProblem->y.push_back(0.55*-2);
    leastSquaresProblem->y.push_back(0.55*-0.6);
    leastSquaresProblem->y.push_back(0.55*0.4);
    leastSquaresProblem->y.push_back(0.55*1.0);
    leastSquaresProblem->y.push_back(0.55*1.6);

    leastSquaresProblem->u = X.size();
    leastSquaresProblem->stdDev.push_back(0.3);
    leastSquaresProblem->stdDev.push_back(0.3);
    leastSquaresProblem->stdDev.push_back(0.3);
    leastSquaresProblem->stdDev.push_back(0.3);
    leastSquaresProblem->stdDev.push_back(0.3);
    leastSquaresProblem->stdDev.push_back(0.3);

    ceres::GradientProblem problem(leastSquaresProblem);
    ceres::GradientProblemSolver::Options options;
    options.line_search_direction_type = ceres::LBFGS;
    options.minimizer_progress_to_stdout = true; 
    ceres::GradientProblemSolver::Summary summary;
    ceres::Solve(options, problem, &X[0], &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";
    for(int n = 0; n < X.size(); n++)
        std::cout<<"after X: "<<X[n]<<std::endl;






    // ceres::Solver::Options options;
    // options.max_num_iterations = 50;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // sparse solver
    // options.minimizer_progress_to_stdout = true;
	// options.max_lm_diagonal = 1.0E-150; // force it behave like a Gauss-Newton update
    // options.min_lm_diagonal = 1.0E-150;
    // ceres::Solver::Summary summary;
    // ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << "\n";
    // std::cout << summary.FullReport() << "\n";

    // for(int n = 0; n < X.size(); n++)
    //     std::cout<<"final X: "<<X[n]<<std::endl;

    // ceres::Covariance::Options covarianceOoptions;
    // ceres::Covariance covariance(covarianceOoptions);

    // std::vector<std::pair<const double*, const double*> > covariance_blocks;
    // for(int i = 0; i < X.size(); i++)
    //     for(int j = 0; j < X.size(); j++)
    //         covariance_blocks.push_back(std::make_pair(&X[i], &X[j]));

    // covariance.Compute(covariance_blocks, &problem);

    // //double covariance_X[X.size() * X.size()];
    // Eigen::Matrix<double,2,2> covariance_X;
    // // covariance_X.resize(X.size() * X.size());
    // //covariance.GetCovarianceBlock(&X[0], &X[0], covariance_X.data());

    // for(int i = 0; i < X.size(); i++)
    //     for(int j = 0; j < X.size(); j++)
    //     {
    //         double temp[1];
    //         temp[0] = 0.0;
    //         covariance.GetCovarianceBlock(&X[i], &X[j], temp);
    //         std::cout<<"std: "<<sqrt(temp[0])<<std::endl;
    //         covariance_X(i,j) = temp[0];
    //     }

    // // std::cout<<"std: "<<sqrt(covariance_X[0])<<", size: "<<covariance_X.size()<<std::endl;
    // // std::cout<<"std: "<<sqrt(covariance_X[4])<<", size: "<<covariance_X.size()<<std::endl;
    //  std::cout << "Variance: " << covariance_X.diagonal() << std::endl; 

    //  std::cout<<"covariance matrix: "<<std::endl;
    //  std::cout<<covariance_X<<std::endl;




    // ceres::Solve(ceres::GradientProblemSolver::Options(options), &ceres::GradientProblem(&problem), &GradientProblemSolver::Summary(summary));
    // ceres::GradientProblem

    // std::cout << summary.BriefReport() << "\n";
    // std::cout << summary.FullReport() << "\n";

    PyRun_SimpleString("print 'building Ceres-Solver cost functions:', round(TIME.clock()-t0, 3), 's' ");

    return 0;
}
