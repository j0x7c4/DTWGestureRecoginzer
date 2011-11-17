#include "DTWGestureRecognizer.h"
#include <fstream>
DTWGestureRecognizer::DTWGestureRecognizer(int dim,int ws,double threshold,double first_threshold) {
  dim_ = dim;
  global_threshold_ = threshold ;
  first_threshold_ = first_threshold;
  window_size_ = ws;
}

DTWGestureRecognizer::DTWGestureRecognizer(string data_file_name, string label_file_name, int ws,double threshold,double first_threshold){
  int dim,m,n;
  double t;
  FILE *fp_data, *fp_label;
  char tc[2];
//parameters
  global_threshold_ = threshold ;
  first_threshold_ = first_threshold;
  window_size_ = ws;
  vector<string> tlabel;

  fp_data = fopen(data_file_name.c_str(),"r");
  if ( !fp_data ) {
    printf("Open data file failed!\n");
    exit(0);
  }
  fp_label = fopen(label_file_name.c_str(),"r");
  if ( !fp_label ) {
    printf("Open data file failed!\n");
    exit(0);
  }

  //read label
  
  
  while ( fscanf(fp_label,"%d,%d",&m,&n)!=EOF ) {
    fgets(tc,2,fp_label);
    for ( int i=0 ; i<m ; i++ ) {
      for ( int j=0 ; j<n ; j++ ) {
        string ts;
        while ( fscanf(fp_label,"%c",&tc) ) {
          if (tc[0]<'0' || tc[0]>'9' ) break;
          ts.push_back(tc[0]);
        }
        tlabel.push_back(ts);
      }
    }
  }

  fclose(fp_label);

  //read data
  int k = 0;
  //fscanf(fp_data,"%d,%d",&dim,&n);
  while ( fscanf(fp_data,"%d,%d",&dim,&n)!=EOF ) {
    fgets(tc,2,fp_label);
    dim_=dim;
    vector<vector<double>> tdata(n,vector<double>()) ;
    
    for ( int i=0 ; i<dim_; i++ ){
      for ( int j=0 ; j<n ; j++ ) {
        fscanf(fp_data,"%lf,",&t);
        //getchar();
        tdata[j].push_back(t);
      }
    }
    AddToGaussianModel(tdata,tlabel[k++]);
  }
  fclose(fp_data);
  
}

void DTWGestureRecognizer::Add(FeatureSequence seq, string l) {
  known_sequences_.push_back(seq);
  known_labels_.push_back(l);
}

void DTWGestureRecognizer::AddToGaussianModel(FeatureSequence seq, string l) {
  int i=0;
  if ( examples_.count(l)>0 ) { //already exisit, update the model
    for ( vector<GaussianModel>::iterator iter = examples_[l].begin(); iter!=examples_[l].end() ; iter++,i++ ) {
      (*iter).Update(seq[i]);
    }
  }
  else { //create a new model
    known_labels_.push_back(l);
    for ( FeatureSequence::iterator iter = seq.begin(); iter!=seq.end() ; iter++ ) {
      GaussianModel gm;
      gm.Init((*iter).size());
      gm.Update(*iter);
      examples_[l].push_back(gm);
    }
  }
}
string DTWGestureRecognizer::Recognize(FeatureSequence seq) {
  double min_dist(INF);
  string _class = "UNKNOWN";
  for(int i=0 ; i<known_sequences_.size() ;i++) {
    FeatureSequence example = known_sequences_[i];
    //if (Euclidian(seq[seq.size()-1], example[seq.size()-1]) < first_threshold_) {
    double d = dtw(seq, example) / (example.size());
    if (d < min_dist){
      min_dist = d;
      _class = (string)(known_labels_[i]);
    }
    //}
  }
  return (min_dist<global_threshold_ ? _class : "UNKNOWN")/*+" "+minDist.ToString()*/ ;
}

string DTWGestureRecognizer::RecognizeByGaussianModel(FeatureSequence seq, double& po) {
  double max_probability(0);
  string _class = "UNKNOWN";
  for(map<string,vector<GaussianModel>>::iterator iter=examples_.begin(); iter!=examples_.end(); iter++) {
    vector<GaussianModel> example = iter->second;  
    double p = dtw(seq, example);
    if (p > max_probability){
      max_probability = p;
      _class = (string)(iter->first);
    }
  }
  po = max_probability;
  printf("%lf ",max_probability);
  return (max_probability>global_threshold_ ? _class : "UNKNOWN")/*+" "+minDist.ToString()*/ ;
}
void DTWGestureRecognizer::ShowGM ( ) {
  for ( Map::iterator i = examples_.begin() ; i!=examples_.end() ; i++ ) {
    for ( GaussianVector::iterator j = i->second.begin() ; j!=i->second.end(); j++ ) {
      cout<<"*************"<<endl;
      j->Show();
    }
    
  }
}
// Computes a 1-distance between two observations. (aka Manhattan distance).
double DTWGestureRecognizer::Manhattan(FeatureData &a, FeatureData &b) {
  double d(0.0);
  for(int i=0 ; i<dim_ ;i++){
    d += fabs(a[i]-b[i]) ;
  }
  return d ;
}

// Computes a 1-distance between two observations. (aka Manhattan distance).
double DTWGestureRecognizer::Euclidian(FeatureData &a, FeatureData &b) {
  double d(0.0);
  for(int i=0 ; i<dim_ ;i++){
    d += pow(a[i]-b[i],2) ;
  }
  return sqrt(d) ;
}

 // Compute the min DTW distance between seq2 and all possible endings of seq1.
double DTWGestureRecognizer::dtw(FeatureSequence seq1, FeatureSequence seq2, int constraint){
  // Init
  int m(
    seq1.size());
  int n(seq2.size());
  double cost;
  vector<vector<double>> tab(m+1,vector<double>(n+1,0.0));

  for(int i=0 ; i<=m ; i++){
    for(int j=0 ; j<=n ; j++){
      tab[i][j]=INF ;
    }
  }
  tab[0][0]=0 ;

  // Dynamic computation of the DTW matrix.
  for(int i=1 ; i<=m ; i++){
    for(int j=MAX(1,i-constraint) ; j<=MIN(n,i+constraint) ; j++){
      cost = Euclidian(seq1[i-1],seq2[j-1]);
      tab[i][j] = cost + MIN(tab[i-1][j-1],MIN(tab[i-1][j],tab[i][j-1]));
    }
  }

  // Find best between seq2 and an ending (postfix) of seq1.
  double best_match = INF;
  for (int i = 1; i <= m; i++) {
    best_match = MIN(best_match,tab[i][n]);
  }
  return best_match;
}

// Compute the max probablity of a model to seq and all possible endings in model.
double DTWGestureRecognizer::dtw(FeatureSequence seq, vector<GaussianModel> example_model,  int constraint){
  // Init
  int n(example_model.size());
  int m(seq.size());
  double p,t;
  vector<vector<double>> tab(m+1,vector<double>(n+1,0.0));

  for(int i=0 ; i<=m ; i++){
    for(int j=0 ; j<=n ; j++){
      tab[i][j]=0 ;
    }
  }
  tab[0][0]=1 ;

  // Dynamic computation of the DTW matrix.
  for(int i=1 ; i<=m ; i++){
    for(int j=MAX(1,i-constraint) ; j<=MIN(n,i+constraint) ; j++){
      p = -log(example_model[j-1].GetProbability(seq[i-1],example_model[j-1].GetCovarianceMatrix(),example_model[j-1].GetMeanVector()));
      t = MAX(tab[i-1][j-1],MAX(tab[i-1][j],tab[i][j-1]));
      tab[i][j] = p + t;
    }
  }

  // Find best between seq and an ending (postfix).
  double best_match = 0;
  for (int i = 1; i <= m; i++) {
    best_match = MAX(best_match,tab[i][n]);
  }
  return best_match;
}
//Get every model trained
int DTWGestureRecognizer::Learning (string label) {
  GaussianFeatureDataList example = examples_[label];
  for (GaussianFeatureDataList::iterator iter_frame = example.begin(); 
    iter_frame!=example.end() ; iter_frame++ ) {
      try { 
        iter_frame->Compute();
      }
      catch (int e) {
        return 1;
      }
  }
  return 0;
}
//Store every model in a file
void DTWGestureRecognizer::Save (FILE* fp, string label) {
  GaussianFeatureDataList seq = examples_[label];
  fprintf(fp,"%d %d\n",dim_,seq.size());
  for (GaussianFeatureDataList::iterator iter_frame = seq.begin(); 
    iter_frame!=seq.end() ; iter_frame++ ) {
      cv::Mat c(iter_frame->GetCovarianceMatrix());
      cv::Mat u(iter_frame->GetMeanVector());
      int dim = iter_frame->GetDim();
      
      //output covariance matrix
      for ( int i=0 ; i<dim ; i++ ) {
        if ( i ) fprintf(fp,"\n");
        for ( int j=0 ; j<dim ; j++ ) {
          if ( j ) fprintf(fp," ");
          fprintf(fp,"%lf",*((double*)c.data+i*dim+j));
        }
      }
      fprintf(fp,"\n");
      //output mean vector
      for ( int i=0 ; i<dim ; i++ ) {
        fprintf(fp,"%lf\n",*((double*)u.data+i));
      }
  }
}

//Show every model on screen
void DTWGestureRecognizer::PrintModel (string label) {
  GaussianFeatureDataList seq = examples_[label];
  printf("Dimension %d, Size %d\n",dim_,seq.size());
  int k=0;
  for (GaussianFeatureDataList::iterator iter_frame = seq.begin(); 
    iter_frame!=seq.end() ; iter_frame++ ) {
      printf("Frame %d:\n",++k);
      cv::Mat c(iter_frame->GetCovarianceMatrix());
      cv::Mat u(iter_frame->GetMeanVector());
      int dim = iter_frame->GetDim();
      //printf("%d\n",dim); //output dimension
      //output covariance matrix
      for ( int i=0 ; i<dim ; i++ ) {
        if ( i ) printf("\n");
        for ( int j=0 ; j<dim ; j++ ) {
          if ( j ) printf(" ");
          printf("%lf",*((double*)c.data+i*dim+j));
        }
      }
      printf("\n");
      //output mean vector
      for ( int i=0 ; i<dim ; i++ ) {
        printf("%lf\n",*((double*)u.data+i));
      }
  }
}