#include "DTWGestureRecognizer.h"
#include <stdio.h>

template <class T>
void convertFromString(T &value, const std::string &s) {
  std::stringstream ss(s);
  ss >> value;
}
int ReadSequence ( FILE* fp, vector<vector<double>>& seq ) {
  char tc[2];
  int dim,n;
  double t;
  if ( fscanf(fp,"%d,%d",&dim,&n)==EOF ) return 0;
  fgets(tc,2,fp);
  seq.clear();
  seq.resize(n,vector<double>());
  for ( int i=0 ; i<dim; i++ ){
    for ( int j=0 ; j<n ; j++ ) {
      fscanf(fp,"%lf,",&t);
      seq[j].push_back(t);
    }
  }
  return 1;
}
int ReadFile ( char* data_file_name, char* label_file_name , DTWGestureRecognizer& dtw ) { 
  int dim,m,n;
  double t;
  FILE *fp_data, *fp_label;
  char tc[2];
//parameters
  vector<string> tlabel;

  fp_data = fopen(data_file_name,"r");
  if ( !fp_data ) {
    printf("Open data file failed!\n");
    return 1;
  }
  fp_label = fopen(label_file_name,"r");
  if ( !fp_label ) {
    printf("Open label file failed!\n");
    return 1;
  }

  //read label
  fscanf(fp_label,"%d,%d",&m,&n);
  //while ( fscanf(fp_label,"%d,%d",&m,&n)!=EOF ) {
  fgets(tc,2,fp_label);
  for ( int i=0 ; i<m ; i++ ) {
    for ( int j=0 ; j<n ; j++ ) {
      string ts;
      while ( fscanf(fp_label,"%c",&tc)!=EOF ) {
        if (tc[0]<'0' || tc[0]>'9' ) break;
        ts.push_back(tc[0]);
      }
      tlabel.push_back(ts);
    }
  }
  //}

  fclose(fp_label);

  //read data
  int k = 0;
  vector<vector<double>> tdata;
  while ( ReadSequence(fp_data,tdata) ) {
    //dtw.AddToGaussianModel(tdata,tlabel[k++]);
    dtw.Add(tdata,tlabel[k++]);
  }
  fclose(fp_data);
  return 0;
}


int main (int argc, char **argv) {
  FILE *fp;
  char data_file[100], label_file[100];
  //parameters
  //-d dimension of feature vector
  //-th threshold
  //-w width of constraint window
  //-fin list file
  //-ft path of test file
  //-fs save path of model data file
  //-fout 
  int dim=-1;
  int width=MAXINT;
  double threshold=0;
  char* fin_name=NULL;
  char* ft_name =NULL;
  char* fsave_name = NULL;
  char* fout_name = NULL;

  for ( int i = 1 ; i<argc ; i+=2 ) {
    string p(argv[i]);
    char* value = argv[i+1];
    if ( p=="-th" ) {
      threshold = atof(value);
    }
    else if ( p=="-d" ) {
      dim=atoi(value);
    }
    else if ( p=="-w" ) {
      width=atoi(value);
    }
    else if ( p=="-fin" ) {
      fin_name = value;
    }
    else if ( p=="-ft" ) {
      ft_name = value;
    }
    else if ( p=="-fs" ) {
      fsave_name = value;
    }
    else if ( p=="-fout" ) {
      fout_name = value;
    }
    else {
      printf("Wrong Parameters!\n");
      exit(1);
    }
  }
  if ( !fin_name ) {
    printf("We need a list file!\n");
    exit(1);
  }
  if ( dim<0 ) {
    printf("We need a list file!\n");
    exit(1);
  }
  
  fp = fopen(fin_name,"r");
  if ( !fp ) {
    printf("Failed to open %s!\n",fin_name);
    exit(1);
  }

  DTWGestureRecognizer dtw(dim,width,threshold);

  while ( fscanf(fp,"%s %s",data_file,label_file)!=EOF ) {
    printf("Adding data %s %s...",data_file,label_file);
    if ( !ReadFile(data_file,label_file,dtw) ) {
      printf("Success!\n");
    }
    else {
      printf("Fail!\n");
    }
  }
  fclose(fp);

  vector<string> labels = dtw.GetLabels();
/*
  printf("***********Training Models************\n");
  for ( int i=0 ; i<labels.size() ; i++ ) {
    printf("Start to training label %s ",labels[i].c_str());
    if ( !dtw.Learning(labels[i]) ) printf("Success!\n");
    else printf("Fail!\n");
  }
  
  
  printf("***********Show Models************\n");
  for ( int i=0 ; i<labels.size() ; i++ ) {
    printf("Label %s\n",labels[i].c_str());
    dtw.PrintModel(labels[i]);
  }
  //save model
  if ( fsave_name ) {
    fp = fopen(fsave_name,"w");
    if ( !fp ) {
      printf("Failed to open %s!\n",fsave_name);
    }
    else {
      for ( int i=0 ; i<labels.size() ; i++ ) {
        fprintf(fp,"%s\n",labels[i].c_str());
        dtw.Save(fp,labels[i]);
      }
      printf("Saved!\n");
      fclose(fp);
    }
  }
*/
  //test
  if ( ft_name ) {
    printf("***********Testing data************\n");
    vector<vector<double>> test_seq;
    fp = fopen(ft_name,"r");
    if ( !fp ) {
      printf("Failed to open %s!\n",ft_name);
    }
    else {
      char test_file[100];
      int k;
      while ( fscanf(fp,"%s",test_file)!=EOF ) {
        FILE *fp_test = fopen(test_file,"r");
        if ( !fp_test ) {
          printf("Failed to open %s!\n",test_file);
          continue;
        }
        int k=1;
        while ( ReadSequence(fp_test,test_seq) ) {
          printf("Testing the %dth data in %s...",k,test_file);
          double p;
          string res;
          //res = dtw.RecognizeByGaussianModel(test_seq,p);
          res = dtw.Recognize(test_seq);
          //printf("%lf %s\n",p,res.c_str());
          printf(" %s\n",res.c_str());
        }
      }
    }
  }
  
  return 0;
}