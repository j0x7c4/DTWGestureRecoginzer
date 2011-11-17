#include "stdafx.h"
#include "GaussianModel.h"

typedef vector<double> FeatureData;
typedef vector<FeatureData> FeatureSequence; 
typedef string LabelData;
typedef vector<LabelData> LabelDataList;
typedef vector<FeatureSequence> FeatureDataList;
typedef vector<GaussianModel> GaussianVector;
typedef GaussianVector GaussianFeatureDataList;
typedef map<string,vector<GaussianModel>> Map;
#define INF MAXLONG
#define NINF MINLONG
#define MAXINT 1000000
#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

class DTWGestureRecognizer {
  // Size of obeservations vectors.
  int dim_;
  // Known sequences
  FeatureDataList known_sequences_;
  // Labels of those known sequences
  LabelDataList known_labels_;
  // Gaussian Model
  Map examples_;
  // Maximum DTW distance between an example and a sequence being classified.
  double global_threshold_ ;

  // Maximum distance between the last observations of each sequence.
  double first_threshold_;

  // Maximum window size.
  int window_size_;

  // Computes a 1-distance between two observations. (aka Manhattan distance).
  double Manhattan(FeatureData &a, FeatureData &b);
  // Computes a 2-distance between two observations. (aka Euclidian distance).
  double Euclidian(FeatureData &a, FeatureData &b);

public:
  DTWGestureRecognizer(int dim=1, int ws=MAXINT, double threshold=0, double firstThreshold=0);
  DTWGestureRecognizer(string,string,int ws=MAXINT, double threshold=0, double firstThreshold=0);
  // Add a seqence with a label to the known sequences library.
  // The gesture MUST start on the first observation of the sequence and end on the last one.
  // Sequences may have different lengths.
  void Add(FeatureSequence seq, string l);
  void AddToGaussianModel(FeatureSequence seq, string l);
  
  // Recognize gesture in the given sequence.
  // It will always assume that the gesture ends on the last observation of that sequence.
  // If the distance between the last observations of each sequence is too great, or if the overall DTW distance between the two sequence is too great, no gesture will be recognized.
  string Recognize(FeatureSequence seq);
  string RecognizeByGaussianModel(FeatureSequence seq,double& p);
  // Compute the min DTW distance between seq2 and all possible endings of seq1.
  double dtw(FeatureSequence seq1, FeatureSequence seq2,int constraint=MAXINT);
  double dtw(FeatureSequence seq,vector<GaussianModel> example_model, int constraint=MAXINT);
  int GetLabelNumber ( ) {
    return known_labels_.size();
  }
  LabelDataList GetLabels () {
    return known_labels_;
  }
  void Save (FILE* fp, string label);
  void PrintModel (string label);
  void ShowGM();
  int Learning(string);
};