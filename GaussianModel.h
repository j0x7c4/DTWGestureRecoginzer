#include "stdafx.h"
#define PI 3.141592653589793
class GaussianModel {
  int dim_;
  //mean vector
  cv::Mat u_;
  //covariance matrix
  cv::Mat c_;
  //x
  vector<cv::Mat> samples_;
public:
  GaussianModel(){};
  GaussianModel(int dim, vector<double> x) {
    cv::Mat sample = cv::Mat(x,true);
    samples_.push_back(sample);
    Init(dim);
  }
  //GaussianModel &operator = (GaussianModel r) {
  //  dim_=r.dim_;
  //  r.u_.copyTo(u_);
  //  r.c_.copyTo(c_);
  //  samples_=vector<cv::Mat>(r.samples_);
  //  return *this;
  //}
  int GetDim() {
    return dim_;
  }
  cv::Mat GetCovarianceMatrix ( ) {
    return c_;
  }
  cv::Mat GetMeanVector ( ) {
    //cv::calcCovarMatrix(&samples_[0],samples_.size(),c_,u_,CV_COVAR_NORMAL);
    return u_;
  }
  void Init(int dim) {
    dim_ = dim;
    u_ = cv::Mat(dim,1,CV_64F);
    c_ = cv::Mat(dim,dim,CV_64F);
  }
  void Update(const vector<double>& x) {
    cv::Mat sample(x,true);
    samples_.push_back(sample);   
  }
  double GetProbability ( const vector<double>& x ) {
    cv::Mat test = cv::Mat(x,true);
    cv::calcCovarMatrix(&samples_[0],samples_.size(),c_,u_,CV_COVAR_NORMAL);
    //cout<<c_<<endl;
    //cout<<u_<<endl;
    double detC = cv::determinant(c_);
    cv::Mat transpose_mat = cv::Mat(dim_,1,CV_64F);
    cv::transpose(test-u_,transpose_mat);
    cv::Mat invert_mat = cv::Mat(dim_,dim_,CV_64F);
    cv::invert(c_,invert_mat);
    cv::Mat temp_mat = transpose_mat*invert_mat*(test-u_);
    double res = 1/(pow(2*PI,dim_/2.0)*pow(detC,1.0/2)) * exp(-0.5*(*(double*)temp_mat.data));
    return res;
  }
  double GetProbability ( const vector<double>& x , cv::Mat covariance, cv::Mat mean ) {
    cv::Mat test = cv::Mat(x,true);
    //cout<<c_<<endl;
    //cout<<u_<<endl;
    double detC = cv::determinant(covariance);
    cv::Mat transpose_mat = cv::Mat(dim_,1,CV_64F);
    cv::transpose(test-mean,transpose_mat);
    cv::Mat invert_mat = cv::Mat(dim_,dim_,CV_64F);
    cv::invert(covariance,invert_mat);
    cv::Mat temp_mat = transpose_mat*invert_mat*(test-mean);
    double res = 1/(pow(2*PI,dim_/2.0)*pow(detC,1.0/2)) * exp(-0.5*(*(double*)temp_mat.data));
    return res;
  }
  
  void Compute() {
    cv::Mat u = cv::Mat(dim_,1,CV_64F);
    cv::Mat c = cv::Mat(dim_,dim_,CV_64F);
    //printf("1:%d\n",u.data);
    cv::calcCovarMatrix(&samples_[0],samples_.size(),c,u,CV_COVAR_NORMAL);
    //printf("2:%d\n",u.data);
    u.copyTo(u_);
    c.copyTo(c_);
  }
  void Show () {
    cv::calcCovarMatrix(&samples_[0],samples_.size(),c_,u_,CV_COVAR_NORMAL);
    cout<<c_<<endl;
    cout<<u_<<endl;
  }
};